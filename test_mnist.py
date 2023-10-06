import torch
from torch import nn
import numpy as np
import realnvp as rnvp
from datasets import set_datasets
from autoencoder import *
from utils import *
import os

from torch.utils.data import DataLoader

import argparse
from argparse import Namespace
from torchvision import utils as vutils
import torchvision.transforms.functional as F
import random


args = parse_arguments()

EMBEDDING_DIM = args.code_length # The dimension of the embeddings
AE_H = args.ae_h
FLOW_N = args.num_blocks # Number of affine coupling layers
RNVP_TOPOLOGY = [args.nf_h] # Size of the hidden layers in each coupling layer
SEED = args.seed # Seed of the random number generator
BATCH_SIZE = args.batch_size # Batch size
ITERATIONS = args.iterations
GRAD_CLIP = args.grad_clip
LOG_STEP = args.log_step
NOISE = args.noise
AE_LR = args.ae_lr
NF_LR = args.nf_lr
W_DECAY = args.weight_decay
CLASS = args.select


# Set the random seeds
set_random_seed(SEED)
print(f'******************** Random seed = {args.seed} *************************')

def _init_fn():
    np.random.seed(12)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the dataset
train_set, val_set, test_set = set_datasets(args.dataset, None)
INPUT_SHAPE = (1, 28, 28)

real_train_loader = DataLoader(train_set, batch_size=200, shuffle=False, worker_init_fn=_init_fn)
if args.test_flag:
    real_test_loader = DataLoader(test_set, batch_size=200, shuffle=False, worker_init_fn=_init_fn)
else:
    real_test_loader = DataLoader(val_set, batch_size=200, shuffle=False, worker_init_fn=_init_fn)


if args.MLACC_flag:
    # Compute ML ACC when training on synthetic data, then test on real data
    real_train = []
    real_test = []
    syn_train = []

    for _, (x, y) in enumerate(real_train_loader):
        x = x.view(x.size(0), -1)  # flatten
        y = torch.unsqueeze(y, 1).float()
        real_train.append(torch.cat([x, y], dim=1))

    for _, (x, y) in enumerate(real_test_loader):
        x = x.view(x.size(0), -1)  # flatten
        y = torch.unsqueeze(y, 1).float()
        real_test.append(torch.cat([x, y], dim=1))

    del y

    for label in range(args.n_class):
        checkpoints_dir = f'./checkpoints/{args.dataset}/DPLFlow-class{label}/h{AE_H}b{FLOW_N}h{args.nf_h}/'
        # Build the autoencoder
        autoencoder = AutoEncoder(INPUT_SHAPE, AE_H, EMBEDDING_DIM)
        autoencoder = autoencoder.to(device)
        AE_path = os.path.join(checkpoints_dir,
                               f'DPAE_{args.num_iters}_gradclip{GRAD_CLIP}_noise{NOISE}_lr{AE_LR}_bs{BATCH_SIZE}.pkl')
        autoencoder.load_state_dict(torch.load(AE_path), strict=False)
        print('AE_path:', AE_path)

        nf_model = rnvp.LinearRNVP(input_dim=EMBEDDING_DIM, coupling_topology=RNVP_TOPOLOGY, flow_n=FLOW_N,
                                   batch_norm=False,
                                   mask_type='odds', conditioning_size=None, use_permutation=True,
                                   single_function=True)
        nf_model = nf_model.to(device)
        NF_path = os.path.join(checkpoints_dir,
                               f'DPNF_{args.num_iters}_gradclip{GRAD_CLIP}_noise{NOISE}_lr{NF_LR}_bs{BATCH_SIZE}.pkl')
        nf_model.load_state_dict(torch.load(NF_path), strict=False)
        print('NF_path:', NF_path)
        for _ in range(int(args.num_samples / args.n_class / 100)):
            emb, d = nf_model.sample(100, return_logdet=True)
            images = autoencoder.decoder(emb)
            images = torch.sigmoid(images).detach().cpu()
            if args.sharpness >= 0:
                images = F.adjust_sharpness(images, 0.0) # smooth first
                images = F.adjust_sharpness(images, args.sharpness)
            images = images.view(images.size(0), -1)
            y = torch.ones(100) * label
            y = torch.unsqueeze(y, 1).float()
            syn_train.append(torch.cat([images, y], dim=1))

    syn_train_loader = DataLoader(torch.cat(syn_train, dim=0), batch_size=200, shuffle=True, worker_init_fn=_init_fn)

    real_train = torch.cat(real_train, dim=0)
    real_test = torch.cat(real_test, dim=0)
    syn_train = torch.cat(syn_train, dim=0)

    evalLR(real_train, real_test, syn_train, args.seed)

    ## NN
    print('==========MLP===========')
    model = NNClassifier(INPUT_SHAPE, type='MLP', data_mode='syn', dataset=args.dataset)
    model = NNfit(model, syn_train_loader, lr=0.01, num_epochs=20, data_mode='syn')
    print('==> syn data:')
    evalNN(model, real_test_loader, data_mode='syn', dataset='mnist')

    print('==========CNN===========')
    model = NNClassifier(INPUT_SHAPE, type='CNN', data_mode='syn', dataset=args.dataset)
    model = NNfit(model, syn_train_loader, lr=0.001, num_epochs=5, data_mode='syn')
    print('==> syn data:')
    evalNN(model, real_test_loader, data_mode='syn', dataset='mnist')
elif args.FID_flag:
    saveImgDir = f'./Imgs/{args.dataset}/'
    genImgDir = f'{saveImgDir}Gen/{args.num_samples}/'
    if args.test_flag:
        trueImgDir = os.path.join(saveImgDir, 'True')
    else:
        trueImgDir = os.path.join(saveImgDir, 'True_val')

    if not os.path.exists(trueImgDir):
        os.makedirs(trueImgDir)
        print(f'Make Dir:{trueImgDir}')

    if not os.path.exists(genImgDir):
        os.makedirs(genImgDir)
        print(f'Make Dir:{genImgDir}')

    if len(os.listdir(trueImgDir)) == 0:
        ii = 0
        for _, (x, y) in enumerate(real_test_loader):
            for xi in x:
                vutils.save_image(xi, f'{trueImgDir}/imgs_{ii}.png')
                ii += 1
        print('Total: ', ii, ' true test images.')

    ii = 0
    for label in range(args.n_class):
        checkpoints_dir = f'./checkpoints/{args.dataset}/DPLFlow-class{label}/h{AE_H}b{FLOW_N}h{args.nf_h}/'
        # Build the autoencoder
        autoencoder = AutoEncoder(INPUT_SHAPE, AE_H, EMBEDDING_DIM)
        autoencoder = autoencoder.to(device)
        AE_path = os.path.join(checkpoints_dir,
                               f'DPAE_{args.num_iters}_gradclip{GRAD_CLIP}_noise{NOISE}_lr{AE_LR}_bs{BATCH_SIZE}.pkl')
        autoencoder.load_state_dict(torch.load(AE_path), strict=False)
        print('AE_path:', AE_path)

        nf_model = rnvp.LinearRNVP(input_dim=EMBEDDING_DIM, coupling_topology=RNVP_TOPOLOGY, flow_n=FLOW_N,
                                   batch_norm=False,
                                   mask_type='odds', conditioning_size=None, use_permutation=True,
                                   single_function=True)
        nf_model = nf_model.to(device)
        NF_path = os.path.join(checkpoints_dir,
                               f'DPNF_{args.num_iters}_gradclip{GRAD_CLIP}_noise{NOISE}_lr{NF_LR}_bs{BATCH_SIZE}.pkl')
        nf_model.load_state_dict(torch.load(NF_path), strict=False)
        print('NF_path:', NF_path)
        for _ in range(int(args.num_samples / args.n_class / 100)):
            emb, d = nf_model.sample(100, return_logdet=True)
            images = autoencoder.decoder(emb)
            images = torch.sigmoid(images).detach().cpu()
            if args.sharpness >= 0:
                images = F.adjust_sharpness(images, 0.0) # smooth first
                images = F.adjust_sharpness(images, args.sharpness)
            for img in images:
                vutils.save_image(img, f'{genImgDir}/imgs_{ii}.png')
                ii += 1
    print('Generated ', ii, ' images.')
elif args.OOD_flag:
    for label in range(args.n_class):
        checkpoints_dir = f'./checkpoints/{args.dataset}/DPLFlow-class{label}/h{AE_H}b{FLOW_N}h{args.nf_h}/'
        # Build the autoencoder
        autoencoder = AutoEncoder(INPUT_SHAPE, AE_H, EMBEDDING_DIM)
        autoencoder = autoencoder.to(device)
        AE_path = os.path.join(checkpoints_dir,
                               f'DPAE_{args.num_iters}_gradclip{GRAD_CLIP}_noise{NOISE}_lr{AE_LR}_bs{BATCH_SIZE}.pkl')
        autoencoder.load_state_dict(torch.load(AE_path), strict=False)
        print('AE_path:', AE_path)

        nf_model = rnvp.LinearRNVP(input_dim=EMBEDDING_DIM, coupling_topology=RNVP_TOPOLOGY, flow_n=FLOW_N,
                                   batch_norm=False,
                                   mask_type='odds', conditioning_size=None, use_permutation=True,
                                   single_function=True)
        nf_model = nf_model.to(device)
        NF_path = os.path.join(checkpoints_dir,
                               f'DPNF_{args.num_iters}_gradclip{GRAD_CLIP}_noise{NOISE}_lr{NF_LR}_bs{BATCH_SIZE}.pkl')
        nf_model.load_state_dict(torch.load(NF_path), strict=False)
        print('NF_path:', NF_path)

        sample_score = torch.Tensor([])
        y_true = torch.Tensor([])
        for _, (x, y) in enumerate(real_test_loader):
            x = x.to(device)
            _, emb = autoencoder(x)
            u, log_det = nf_model.forward(emb)
            prior_logprob = nf_model.logprob(u)
            nll = (prior_logprob.sum(1) + log_det) * (-1.0) # negative log-likelihood, higher implies OOD
            sample_score = torch.cat([sample_score, nll.detach().cpu()], dim=0)
            y_true = torch.cat([y_true, (y != label) * 1.0], dim=0) # 1 means OOD, 0 means InD

        sample_score = sample_score.numpy()
        y_true = y_true.numpy()
        print(f'>>> Intra-dataset OOD detection on {args.dataset} for class {label}:')
        ood_auc(sample_score, y_true)

else:
    # generating aggregated images
    images_all = torch.Tensor([])
    for label in range(args.n_class):
        checkpoints_dir = f'./checkpoints/{args.dataset}/DPLFlow-class{label}/h{AE_H}b{FLOW_N}h{args.nf_h}/'
        # Build the autoencoder
        autoencoder = AutoEncoder(INPUT_SHAPE, AE_H, EMBEDDING_DIM)
        autoencoder = autoencoder.to(device)
        AE_path = os.path.join(checkpoints_dir, f'DPAE_{args.num_iters}_gradclip{GRAD_CLIP}_noise{NOISE}_lr{AE_LR}_bs{BATCH_SIZE}.pkl')
        autoencoder.load_state_dict(torch.load(AE_path), strict=False)
        print('AE_path:', AE_path)

        nf_model = rnvp.LinearRNVP(input_dim=EMBEDDING_DIM, coupling_topology=RNVP_TOPOLOGY, flow_n=FLOW_N,
                                   batch_norm=False,
                                   mask_type='odds', conditioning_size=None, use_permutation=True, single_function=True)
        nf_model = nf_model.to(device)
        NF_path = os.path.join(checkpoints_dir, f'DPNF_{args.num_iters}_gradclip{GRAD_CLIP}_noise{NOISE}_lr{NF_LR}_bs{BATCH_SIZE}.pkl')
        nf_model.load_state_dict(torch.load(NF_path), strict=False)
        print('NF_path:', NF_path)

        emb, d = nf_model.sample(8, return_logdet=True)
        images = autoencoder.decoder(emb)
        images = torch.sigmoid(images).cpu()
        if args.sharpness >= 0:
            images = F.adjust_sharpness(images, 0.0)  # smooth first
            images = F.adjust_sharpness(images, args.sharpness)
        images_all = torch.cat([images_all, images], dim=0)

    images_rearraged = torch.randn_like(images_all)
    idx = 0
    for i in range(8):
        for label in range(args.n_class):
            images_rearraged[idx] = images_all[label * 8 + i]
            idx += 1
    grid = vutils.make_grid(images_rearraged, nrow=10)
    vutils.save_image(grid, f'DPAENF_{args.dataset}_iter{args.num_iters}_gradclip{args.grad_clip}_noise{args.noise}_'
                            f'aelr{AE_LR}_nflr{NF_LR}_bs{BATCH_SIZE}_h{AE_H}n{FLOW_N}h{args.nf_h}_'
                            f'sharpness{args.sharpness}.png')