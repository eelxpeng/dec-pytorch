import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import argparse
from lib.stackedDAE import StackedDAE
from lib.dec import DEC
from lib.datasets import MNIST

dataset = "mnist"
repeat = 10
batch_size = 256

for i in range(1, repeat+1):
    print("Experiment #%d" % i)

    train_loader = torch.utils.data.DataLoader(
        MNIST('./dataset/mnist', train=True, download=True),
        batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        MNIST('./dataset/mnist', train=False),
        batch_size=batch_size, shuffle=False, num_workers=0)
    # pretrain
    sdae = StackedDAE(input_dim=784, z_dim=10, binary=False,
        encodeLayer=[500,500,2000], decodeLayer=[2000,500,500], activation="relu", 
        dropout=0)
    print(sdae)
    sdae.pretrain(train_loader, test_loader, lr=0.1, batch_size=batch_size, 
        num_epochs=300, corrupt=0.2, loss_type="mse")
    sdae.fit(train_loader, test_loader, lr=0.1, num_epochs=500, corrupt=0.2, loss_type="mse")
    sdae_savepath = ("model/sdae-run-%d.pt" % i)
    sdae.save_model(sdae_savepath)

    # finetune
    mnist_train = MNIST('./dataset/mnist', train=True, download=True)
    mnist_test = MNIST('./dataset/mnist', train=False)
    X = mnist_train.train_data
    y = mnist_train.train_labels

    dec = DEC(input_dim=784, z_dim=10, n_clusters=10,
        encodeLayer=[500,500,2000], activation="relu", dropout=0)
    print(dec)
    dec.load_model(sdae_savepath)
    dec.fit(X, y, lr=0.01, batch_size=256, num_epochs=100, 
        update_interval=1)
    dec_savepath = ("model/dec-run-%d.pt" % i)
    dec.save_model(dec_savepath)