#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
import numpy as np
from sklearn.linear_model import LinearRegression


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:#独立同分布的设置
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:#非独立同分布
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)
    # print(type(user_groups))

    return train_dataset, test_dataset, user_groups


def cosine_similarity_score(local_weight, global_weight):
    """
    计算分数（余弦相似度）
    args:
    local_weight (dict): A single client's model weights.
    global_weight (dict): The global model weights.

    Returns:
    float: The cosine similarity score.
    """
    # Flatten the weights to compute cosine similarity
    local_weight_flat = torch.cat([w.view(-1) for w in local_weight.values()])
    global_weight_flat = torch.cat([w.view(-1) for w in global_weight.values()])

    # Compute cosine similarity
    score = F.cosine_similarity(local_weight_flat.unsqueeze(0), global_weight_flat.unsqueeze(0))

    return score.item()

# 创建并拟合线性回归模型
def CreateLinearRegression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    predict_xth = model.predict([[len(X)+1]])
    return len(X)+1,predict_xth

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
        print(f'    num of users  : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
