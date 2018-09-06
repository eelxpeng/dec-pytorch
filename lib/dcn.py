import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import math
from lib.utils import Dataset, masking_noise
from lib.ops import MSELoss, BCELoss
from lib.denoisingAutoencoder import DenoisingAutoencoder
from lib.utils import acc
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans

def buildNetwork(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)

def batch_km(data, center, count):
    """
    Function to perform a KMeans update on a batch of data, center is the
    centroid from last iteration.

    """
    N = data.shape[0]
    K = center.shape[0]

    # update assignment
    idx = np.zeros(N, dtype=np.int)
    for i in range(N):
        dist = np.inf
        ind = 0
        for j in range(K):
            temp_dist = np.linalg.norm(data[i] - center[j])
            if temp_dist < dist:
                dist = temp_dist
                ind = j
        idx[i] = ind

    # update centriod
    center_new = center
    for i in range(N):
        c = idx[i]
        count[c] += 1
        eta = 1.0/count[c]
        center_new[c] = (1 - eta) * center_new[c] + eta * data[i]
    center_new.astype(np.float32)
    return idx, center_new, count

class DeepClusteringNetwork(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, n_centroids=10, binary=True,
        encodeLayer=[400], decodeLayer=[400], activation="relu", 
        dropout=0, tied=False):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.layers = [input_dim] + encodeLayer + [z_dim]
        self.activation = activation
        self.dropout = dropout
        self.encoder = buildNetwork([input_dim] + encodeLayer, activation=activation, dropout=dropout)
        self.decoder = buildNetwork([z_dim] + decodeLayer, activation=activation, dropout=dropout)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        
        self._dec = nn.Linear(decodeLayer[-1], input_dim)
        self._dec_act = None
        if binary:
            self._dec_act = nn.Sigmoid()

    def decode(self, z):
        h = self.decoder(z)
        x = self._dec(h)
        if self._dec_act is not None:
            x = self._dec_act(x)
        return x

    def loss_function(self, recon_x, x, z, center):
        if self._dec_act is not None:
          recon_loss = -torch.mean(torch.sum(x*torch.log(torch.clamp(recon_x, min=1e-10))+
              (1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1))
        else:
          recon_loss = torch.mean(torch.sum((x - recon_x)**2, 1))

        cluster_loss = torch.mean(torch.sum((center - z)**2, 1))
        loss = cluster_loss + recon_loss
        return loss, recon_loss, cluster_loss

    def forward(self, x):
        h = self.encoder(x)
        z = self._enc_mu(h)

        return z, self.decode(z)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def pretrain(self, trainloader, validloader, lr=0.001, batch_size=128, num_epochs=10, corrupt=0.3, loss_type="mse"):
        trloader = trainloader
        valoader = validloader
        daeLayers = []
        for l in range(1, len(self.layers)):
            infeatures = self.layers[l-1]
            outfeatures = self.layers[l]
            dae = DenoisingAutoencoder(infeatures, outfeatures, activation=self.activation, dropout=self.dropout, tied=True)
            if l==1:
                dae.fit(trloader, valoader, lr=lr, batch_size=batch_size, num_epochs=num_epochs, corrupt=corrupt, loss_type=loss_type)
            else:
                if self.activation=="sigmoid":
                    dae.fit(trloader, valoader, lr=lr, batch_size=batch_size, num_epochs=num_epochs, corrupt=corrupt, loss_type="cross-entropy")
                else:
                    dae.fit(trloader, valoader, lr=lr, batch_size=batch_size, num_epochs=num_epochs, corrupt=corrupt, loss_type="mse")
            data_x = dae.encodeBatch(trloader)
            valid_x = dae.encodeBatch(valoader)
            trainset = Dataset(data_x, data_x)
            trloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True, num_workers=2)
            validset = Dataset(valid_x, valid_x)
            valoader = torch.utils.data.DataLoader(
                validset, batch_size=1000, shuffle=False, num_workers=2)
            daeLayers.append(dae)

        self.copyParam(daeLayers)

    def copyParam(self, daeLayers):
        if self.dropout==0:
            every = 2
        else:
            every = 3
        # input layer
        # copy encoder weight
        self.encoder[0].weight.data.copy_(daeLayers[0].weight.data)
        self.encoder[0].bias.data.copy_(daeLayers[0].bias.data)
        self._dec.weight.data.copy_(daeLayers[0].deweight.data)
        self._dec.bias.data.copy_(daeLayers[0].vbias.data)

        for l in range(1, len(self.layers)-2):
            # copy encoder weight
            self.encoder[l*every].weight.data.copy_(daeLayers[l].weight.data)
            self.encoder[l*every].bias.data.copy_(daeLayers[l].bias.data)

            # copy decoder weight
            self.decoder[-(l-1)*every-2].weight.data.copy_(daeLayers[l].deweight.data)
            self.decoder[-(l-1)*every-2].bias.data.copy_(daeLayers[l].vbias.data)

        # z layer
        self._enc_mu.weight.data.copy_(daeLayers[-1].weight.data)
        self._enc_mu.bias.data.copy_(daeLayers[-1].bias.data)
        self.decoder[0].weight.data.copy_(daeLayers[-1].deweight.data)
        self.decoder[0].bias.data.copy_(daeLayers[-1].vbias.data)

    def encodeBatch(self, data):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
            data = data.cuda()
        z, _ = self.forward(data)
        return z.data.cpu()

    def initialize_cluster(self, trainX, trainY, init="k-means++"):
        trainX = self.encodeBatch(trainX)
        trainX = trainX.cpu().numpy()
        trainY = trainY.cpu().numpy()
        n_components = len(np.unique(trainY))
        km = KMeans(n_clusters=n_components, init=init).fit(trainX)
        y_pred = km.predict(trainX)
        print("acc: %.5f, nmi: %.5f" % (acc(trainY, y_pred), normalized_mutual_info_score(trainY, y_pred)))
        
        u_p = km.cluster_centers_
        return u_p, y_pred

    def fit(self, trainX, trainY, lr=0.001, batch_size=128, num_epochs=10):
        n_components = len(np.unique(trainY))
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("=====Initialize Cluster Centers=======")
        centers, assignments = self.initialize_cluster(trainX, trainY)

        print("=====Stacked Denoising Autoencoding layer=======")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        num_train = trainX.shape[0]
        n_batches = int(math.ceil(num_train / batch_size))
        count = 100*np.ones(n_components, dtype=np.int)
        for epoch in range(num_epochs):
            # train 1 epoch
            train_loss = 0.0
            train_recon_loss = 0.0
            train_cluster_loss = 0.0
            for batch_idx in range(n_batches):
                inputs = trainX[batch_idx*batch_size : min((batch_idx+1)*batch_size, num_train)]
                labels = assignments[batch_idx*batch_size : min((batch_idx+1)*batch_size, num_train)]
                inputs = inputs.view(inputs.size(0), -1).float()
                centers_batch_tensor = torch.from_numpy(centers[labels])
                if use_cuda:
                    inputs = inputs.cuda()
                    centers_batch_tensor = centers_batch_tensor.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)
                centers_batch_tensor = Variable(centers_batch_tensor)

                z, outputs = self.forward(inputs)
                loss, recon_loss, cluster_loss = self.loss_function(outputs, inputs, z, centers_batch_tensor)
                train_loss += loss.data*len(inputs)
                train_recon_loss += recon_loss.data*len(inputs)
                train_cluster_loss += cluster_loss.data*len(inputs)
                loss.backward()
                optimizer.step()

                # Perform mini-batch KM
                temp_idx, centers, count = batch_km(z.data.cpu().numpy(), centers, count)
                assignments[batch_idx*batch_size : min((batch_idx+1)*batch_size, num_train)] = temp_idx

            print("#Epoch %3d: Loss: %.3f, Recon Loss: %.3f, Cluster Loss: %.3f" % (
                epoch+1, train_loss / num_train, train_recon_loss/num_train, train_cluster_loss/num_train))

            if (epoch+1) % 10 == 0:
              centers, assignments = self.initialize_cluster(trainX, trainY, centers)


