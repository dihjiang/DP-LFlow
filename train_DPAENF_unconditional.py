import torch
from torch import nn
import realnvp as rnvp
from autoencoder import *
from utils import *
from datasets import set_datasets
import numpy as np
import os

from pyvacy import sampling, analysis
from pyvacy import optim as opt
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

import argparse
from argparse import Namespace
from torchvision import utils as vutils



args = parse_arguments()

EMBEDDING_DIM = args.code_length # The dimension of the embeddings
AE_H = args.ae_h
FLOW_N = args.num_blocks # Number of affine coupling layers
RNVP_TOPOLOGY = [args.nf_h] # Size of the hidden layers in each coupling layer
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
set_random_seed(args.seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

checkpoints_dir = f'./checkpoints/{args.dataset}/DPLFlow-class{CLASS}/h{AE_H}b{FLOW_N}h{args.nf_h}'

if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

def _init_fn():
    np.random.seed(12)

train_set, _, _, indices_full = set_datasets(args.dataset, args.select)

# Build the autoencoder
if args.dataset in ['mnist', 'fmnist']:
    INPUT_SHAPE = (1, 28, 28)
    autoencoder = AutoEncoder(INPUT_SHAPE, AE_H, EMBEDDING_DIM)
elif args.dataset == 'celeba':
    INPUT_SHAPE = (3, 32, 32)
    autoencoder = AutoEncoder_RGB32(INPUT_SHAPE, AE_H, EMBEDDING_DIM)
elif args.dataset in ['celeba256', 'celebahq']:
    INPUT_SHAPE = (3, 256, 256)
    autoencoder = AutoEncoder_RGB256(INPUT_SHAPE, AE_H, EMBEDDING_DIM)

autoencoder = autoencoder.to(device)

# Reconstruction loss
loss_f = nn.BCELoss()

DPoptimizer = opt.DPAdam(params=autoencoder.parameters(), lr=AE_LR, weight_decay=W_DECAY,
                                            l2_norm_clip=GRAD_CLIP,
                                            noise_multiplier=NOISE,
                                            minibatch_size=BATCH_SIZE,
                                            microbatch_size=1)

# now only need unconditional flow
nf_model = rnvp.LinearRNVP(input_dim=EMBEDDING_DIM, coupling_topology=RNVP_TOPOLOGY, flow_n=FLOW_N, batch_norm=False,
                           mask_type='odds', conditioning_size=None, use_permutation=True, single_function=True)
nf_model = nf_model.to(device)
DPoptimizer1 = opt.DPAdam(params=nf_model.parameters(), lr=NF_LR, weight_decay=W_DECAY,
                                            l2_norm_clip=GRAD_CLIP,
                                            noise_multiplier=NOISE,
                                            minibatch_size=BATCH_SIZE,
                                            microbatch_size=1)

print('#AE parameters:', sum(p.numel() for p in autoencoder.parameters()))
print('#trainable AE parameters:', sum(p.numel() for p in autoencoder.parameters() if p.requires_grad))
print('#NF parameters:', sum(p.numel() for p in nf_model.parameters()))
print('#trainable NF parameters:', sum(p.numel() for p in nf_model.parameters() if p.requires_grad))


trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, drop_last=False,
                         worker_init_fn=_init_fn, sampler=SubsetRandomSampler(indices_full))
input_data = inf_train_gen(trainloader)

_, microbatch_loader = sampling.get_data_loaders(
    BATCH_SIZE,
    1,  # mbs
    ITERATIONS) # iterations

eps = analysis.epsilon(len(indices_full), BATCH_SIZE, NOISE, ITERATIONS)
print(f'We are targeting ({eps}, 1e-5)-DP.')

autoencoder.train()
nf_model.train()

for iter in range(ITERATIONS):
    #########################
    ### Update network
    #########################
    iter_loss = 0.
    iter_recloss = 0.
    iter_nll = 0.
    x, y = next(input_data)

    DPoptimizer1.zero_grad()
    DPoptimizer.zero_grad()
    for X_microbatch, y_microbatch in microbatch_loader(TensorDataset(x, y)):
        X_microbatch = X_microbatch.to(device)
        DPoptimizer1.zero_microbatch_grad()
        DPoptimizer.zero_microbatch_grad()

        x_r, emb = autoencoder(X_microbatch)
        x_r = torch.sigmoid(x_r)
        u, log_det = nf_model.forward(emb)
        prior_logprob = nf_model.logprob(u)
        log_prob = -torch.mean(prior_logprob.sum(1) + log_det)

        # Compute loss
        rec_loss = loss_f(x_r, X_microbatch) * 100000.0
        tot_loss = rec_loss + log_prob
        tot_loss.backward()

        DPoptimizer1.microbatch_step()
        DPoptimizer.microbatch_step()

        iter_loss += tot_loss.item() * X_microbatch.shape[0]
        iter_nll += log_prob.item() * X_microbatch.shape[0]
        iter_recloss += rec_loss.item() * X_microbatch.shape[0]

    DPoptimizer1.step()
    DPoptimizer.step()


    if iter % LOG_STEP == 0:
        # training loss
        print('Current iter: ', iter, 'Total training iters: ', ITERATIONS)
        print('Training loss: {}\tRec_loss:{}\tNLL:{}\tLoss:{:.4f}\t'.format(iter, iter_recloss / x.shape[0],
                                                                             iter_nll / x.shape[0], iter_loss / x.shape[0]))
        torch.save(autoencoder.state_dict(), os.path.join(checkpoints_dir, f'DPAE_{iter}_gradclip{GRAD_CLIP}_noise{NOISE}_lr{AE_LR}_bs{BATCH_SIZE}.pkl'))
        torch.save(nf_model.state_dict(), os.path.join(checkpoints_dir, f'DPNF_{iter}_gradclip{GRAD_CLIP}_noise{NOISE}_lr{NF_LR}_bs{BATCH_SIZE}.pkl'))
        emb, d = nf_model.sample(80, return_logdet=True)
        images = autoencoder.decoder(emb)
        images = torch.sigmoid(images).cpu()
        grid = vutils.make_grid(images, nrow=10)
        vutils.save_image(grid, os.path.join(checkpoints_dir, f'samples_DPAE_DPNF_iter{iter}_aelr{AE_LR}_'
                                                             f'nflr{NF_LR}_gradclip{GRAD_CLIP}_noise{NOISE}_bs{BATCH_SIZE}.png'))
