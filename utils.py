import torch
from torch import nn
import numpy as np
import os

import argparse
from argparse import Namespace
import random
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import roc_auc_score


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--grad_clip', type=float, default=-1.0, help='Gradient norm clip')
    parser.add_argument('--iterations', type=int, default=200, help='#iters')
    parser.add_argument('--log_step', type=int, default=50, help='#log steps')
    parser.add_argument('--epochs', type=int, default=20, help='#training epochs')
    parser.add_argument('--batch_size', type=int, default=50, help='#bs')
    parser.add_argument('--noise', type=float, default=0.61, help='noise multiplier')
    parser.add_argument('--ae_lr', type=float, default=1e-3, help='lr for ae')
    parser.add_argument('--nf_lr', type=float, default=1e-4, help='lr for nf')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--select', type=int, default=0, help='select a class (0-9) for mnist, (0-1) for celeba')
    parser.add_argument('--num_samples', type=int, default=60000, help='number of generated images for computing FID')
    parser.add_argument('--compute_FID', dest='FID_flag', action='store_true', default=False, help='generate images to calculate FID')
    parser.add_argument('--ML_ACC', dest='MLACC_flag', action='store_true', default=False,
                        help='generate images to calculate ML ACC')
    parser.add_argument('--OOD', dest='OOD_flag', action='store_true', default=False,
                        help='Intra-dataset OOD detection')
    parser.add_argument('--n_class', type=int, default=10, help='#class')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--num_iters', type=int, default=200000, help='#iters')
    parser.add_argument('--num_epochs', type=int, default=50, help='#epochs')
    parser.add_argument('--dataset', type=str,
                        help='The name of the dataset to perform tests on.Choose among `mnist`, `fmnist`, `celeba`',
                        metavar='')
    parser.add_argument('--code_length', type=int, default=20, help='code_length')
    parser.add_argument('--cnn_epochs', type=int, default=100, help='cnn epochs')
    parser.add_argument('--ae_h', type=int, default=64, help='#hidden size in the autoencoder')
    parser.add_argument('--nf_h', type=int, default=200, help='#hidden size in the flow')
    parser.add_argument('--num_blocks', type=int, default=9, help='#blocks of flow')
    parser.add_argument('--BN', dest='BN_flag', action='store_true', default=False,
                        help='use batch norm or not in the realnvp')
    parser.add_argument('--DP', dest='dp_flag', action='store_true', default=False,
                        help='load dp trained model')
    parser.add_argument('--test', dest='test_flag', action='store_true', default=False,
                        help='enabled for test; unenabled for validation')
    parser.add_argument('--sharpness', type=float, default=-1.0, help='sharpness for computing FID')
    return parser.parse_args()


def inf_train_gen(trainloader):
    while True:
        for images, targets in trainloader:
            yield (images, targets)

def set_random_seed(seed):
    """
    Sets random seeds.

    :param seed: the seed to be set for all libraries.
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # np.random.shuffle.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NNClassifier(nn.Module):
    def __init__(self, input_shape, type='MLP', data_mode='real', dataset='mnist'):
        super(NNClassifier, self).__init__()

        self.input_shape = input_shape
        self.input_channel, h, w = input_shape
        self.input_dim = self.input_channel * h * w
        self.type = type
        self.data_mode = data_mode
        if 'mnist' in dataset or 'fmnist' in dataset:
            self.output_dim = 10
        elif 'celeba' in dataset:
            self.output_dim = 2

        self.MLP = nn.Sequential(
            nn.Linear(self.input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, self.output_dim),
            nn.Softmax()
        )
        self.flatten_dim = 64 * int(h / 4) * int(w / 4)
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channel, 32, kernel_size=3, stride = 2, padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride = 2, padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            nn.Linear(self.flatten_dim, self.output_dim),
            nn.Softmax()
        )

    def forward(self, x):
        if self.type == 'MLP':
            x = x.view(-1, self.input_dim)
            x = self.MLP(x)
        elif self.type == 'CNN':
            x = x.view(-1, *self.input_shape)
            x = self.conv(x)
            x = x.view(-1, self.flatten_dim)
            x = self.linear(x)
        return x

def NNfit(model, train_loader, lr=0.01, num_epochs=5, data_mode='real'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_func = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for batch_idx, data_true in enumerate(train_loader):
            if data_mode == 'real':
                X, y_true = data_true
            elif data_mode == 'syn':
                X = data_true[:, :-1]
                y_true = data_true[:, -1]
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_func(y_pred, y_true.to(torch.long))
            loss.backward()
            optimizer.step()
            # Total correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            correct += (predicted == y_true.to(torch.long)).sum()
            total += X.size(0)
            # print(correct)
        print('Epoch : {} \tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
            epoch, loss.item(), float(correct * 100) / total)
        )
    return model

def evalNN(model, test_loader, data_mode='real', dataset='mnist'):
    model.eval()
    correct = 0
    total = 0
    for batch_idx, data in enumerate(test_loader):
        if dataset in ['mnist', 'fmnist']:
            X, y_true = data
            if data_mode == 'syn':
                X = X.view(X.size(0), -1)
        elif dataset in ['celeba']:
            X = data[:, :-1]
            y_true = data[:, -1]
        y_pred = model(X)
        # Total correct predictions
        predicted = torch.max(y_pred.data, 1)[1]
        correct += (predicted == y_true.to(torch.long)).sum()
        total += X.size(0)

    print("Test accuracy:{:.3f}% ".format(float(correct * 100) / total))

def evalLR(real_train, real_test, samples, seed, t1='real', t2='syn'):
    real_test = real_test.numpy()
    numTest = real_test.shape[0]
    # Build classifiers
    samples = samples.numpy()
    LRsyn = LogisticRegression(random_state=seed).fit(samples[:, :-1], samples[:, -1])
    print(f"ACC of LR{t2}:", np.sum(LRsyn.predict(real_test[:, :-1]) == real_test[:, -1]) / numTest)
    if real_train is not None:
        real_train = real_train.numpy()
        LRreal = LogisticRegression(random_state=seed).fit(real_train[:, :-1], real_train[:, -1])
        print(f"ACC of LR{t1}:", np.sum(LRreal.predict(real_test[:, :-1]) == real_test[:, -1]) / numTest)

def evalSVM(real_train, real_test, samples, seed, t1='real', t2='syn'):
    real_test = real_test.numpy()
    numTest = real_test.shape[0]
    # Build classifiers
    samples = samples.numpy()
    SVMsyn = svm.SVC(random_state=seed).fit(samples[:, :-1], samples[:, -1])
    print(f"ACC of SVM{t2}:", np.sum(SVMsyn.predict(real_test[:, :-1]) == real_test[:, -1]) / numTest)
    if real_train is not None:
        real_train = real_train.numpy()
        SVMreal = svm.SVC(random_state=seed).fit(real_train[:, :-1], real_train[:, -1])
        print(f"ACC of SVM{t1}:", np.sum(SVMreal.predict(real_test[:, :-1]) == real_test[:, -1]) / numTest)

def ood_auc(sample_score, y_true):
    sample_score = (sample_score - np.min(sample_score)) / (np.max(sample_score) - np.min(sample_score))
    AUC = roc_auc_score(y_true, sample_score)
    print('>>> AUROC: ', AUC)
