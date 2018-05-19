import sys
sys.path.append("..")
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import argparse
from lib.stackedDAE import StackedDAE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--lr', type=float, default=0.1, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--pretrainepochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    args = parser.parse_args()
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./dataset/mnist', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./dataset/mnist', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=False, num_workers=2)

    sdae = StackedDAE(input_dim=784, z_dim=10, binary=True,
        encodeLayer=[500,500,2000], decodeLayer=[2000,500,500], activation="relu", 
        dropout=0)
    print(sdae)
    sdae.pretrain(train_loader, test_loader, lr=args.lr, batch_size=args.batch_size, 
        num_epochs=args.pretrainepochs, corrupt=0.3, loss_type="mse")
    sdae.save_model("model/sdae.pt")
    sdae.fit(train_loader, test_loader, lr=args.lr, num_epochs=args.epochs, corrupt=0.2, loss_type="mse")
