import sys
sys.path.append("..")
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import argparse
from lib.idec import IDEC
from lib.datasets import MNIST

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--update-interval', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--pretrain', type=str, default="", metavar='N',
                    help='number of epochs to train (default: 10)')
    args = parser.parse_args()
    
    # according to the released code, mnist data is multiplied by 0.02
    # 255*0.02 = 5.1. transforms.ToTensor() coverts 255 -> 1.0
    # so add a customized Scale transform to multiple by 5.1
    mnist_train = MNIST('./dataset/mnist', train=True, download=True)
    mnist_test = MNIST('./dataset/mnist', train=False)
    X = mnist_train.train_data
    y = mnist_train.train_labels

    idec = IDEC(input_dim=784, z_dim=10, n_clusters=10,
        encodeLayer=[500,500,2000], decodeLayer=[2000,500,500], activation="relu", dropout=0)
    print(idec)
    idec.load_model(args.pretrain)
    idec.fit(X, y, lr=args.lr, batch_size=args.batch_size, num_epochs=args.epochs, 
        update_interval=args.update_interval)
    idec.save_model("model/idec.pt")
