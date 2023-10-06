import numpy as np
import torch
from torchvision import datasets, transforms
import copy
import os


# Load the dataset
def set_datasets(dataset, CLASS):
    if dataset in ['mnist', 'fmnist']:
        if dataset == 'mnist':
            train_set_all = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
            test_set = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
        elif dataset == 'fmnist':
            train_set_all = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor())
            test_set = datasets.FashionMNIST('./data', train=False, download=True, transform=transforms.ToTensor())

        indices_all = np.arange(len(train_set_all))
        np.random.shuffle(indices_all)

        train_idx = [False] * 60000
        num_train = [0] * 10
        for ind in indices_all:
            label = train_set_all.targets[ind]
            if num_train[label] < 5000:
                train_idx[ind] = True
                num_train[label] += 1

        val_idx = [not idx for idx in train_idx]

        train_set = copy.deepcopy(train_set_all)
        train_set.targets = train_set.targets[train_idx]
        train_set.data = train_set.data[train_idx]
        val_set = copy.deepcopy(train_set_all)
        val_set.targets = val_set.targets[val_idx]
        val_set.data = val_set.data[val_idx]

        if CLASS is not None:
            idx = train_set.targets == CLASS
            train_set.targets = train_set.targets[idx]
            train_set.data = train_set.data[idx]
            print(f'len(train_set) after selecting class {CLASS}: {len(train_set)}')
            indices_full = np.arange(len(train_set))  # for CLASS

    elif dataset in ['celeba']:
        celeba_dist = [50000] * 2
        resize_dim = 32
        train_set = datasets.CelebA('./data/CELEBA', split='train', download=True,
                                           transform=transforms.Compose(
                                               [transforms.CenterCrop(178),
                                                transforms.Resize(resize_dim),
                                                transforms.ToTensor()])
                                           )
        val_set = datasets.CelebA('./data/CELEBA', split='valid', download=True,
                                   transform=transforms.Compose(
                                       [transforms.CenterCrop(178),
                                        transforms.Resize(resize_dim),
                                        transforms.ToTensor()])
                                   )
        test_set = datasets.CelebA('./data/CELEBA', split='test', download=True,
                                   transform=transforms.Compose(
                                       [transforms.CenterCrop(178),
                                        transforms.Resize(resize_dim),
                                        transforms.ToTensor()])
                                   )
        if CLASS is not None:
            # filter male or female, by attribute 20
            indices_full = []
            for i in range(len(train_set)):
                if train_set[i][1][20] == CLASS:
                    indices_full.append(i)
                if len(indices_full) == celeba_dist[CLASS]:
                    break
            indices_full = np.array(indices_full)

    elif dataset in ['celebahq']:
        # Download the dataset by following https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch/blob/main/Face_Gender_Classification_Test_with_CelebA_HQ.ipynb
        resize_dim = 256
        data_dir = './data/CelebAHQ'
        train_set_all = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transforms.Compose(
                                                         [transforms.Resize(resize_dim),
                                                          transforms.ToTensor()])
                                             )
        test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transforms.Compose(
                                                         [transforms.Resize(resize_dim),
                                                          transforms.ToTensor()])
                                             )

        indices_all = np.arange(len(train_set_all))
        np.random.shuffle(indices_all)

        num_train = [14000, 8000]
        train_set = []
        val_set = []
        for ind in indices_all:
            label = train_set_all[ind][1]
            if num_train[label] > 0:
                train_set.append(train_set_all[ind])
                num_train[label] -= 1
            else:
                val_set.append(train_set_all[ind])

        del train_set_all

        num_class_train = [0, 0]
        for data in train_set:
            num_class_train[data[1]] += 1
        print('# samples for each class in the train_set:', num_class_train)
        print(len(train_set), len(val_set), len(test_set)) # 22000 1999 6001

        if CLASS is not None:
            train_set = [train_data for train_data in train_set if train_data[1]==CLASS]
            print(f'len(train_set) after selecting class {CLASS}: {len(train_set)}')
            indices_full = np.arange(len(train_set))  # for CLASS

    if CLASS is not None:
        return train_set, val_set, test_set, indices_full
    else:
        return train_set, val_set, test_set


