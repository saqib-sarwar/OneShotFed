import copy
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC, PMatDiag, PVector

from torch.utils.tensorboard import SummaryWriter
  
import os
import sys
# sys.path.insert(0, '/data/saqib/bayesian_fl/normal_fl/optimizers')

# from ivon import IVON
import ivon

class LocalUpdate(object):
    def __init__(self, args, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
        logdir = '/data/saqib/bayesian_fl/oneshot/logs'
        self.writer = SummaryWriter(log_dir=logdir)


    def train_and_compute_fisher(self, net, i, n_c):            # i is the client index
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args['eta'], momentum = 0.9)
        step_count = 0

        
        for epoch in range(self.args['local_epochs']):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args['device']), labels.to(self.args['device'])
                if(self.args['augmentation']==True):
                    images = self.transform_train(images)
                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                self.writer.add_scalar("Client " + str(i) + " Loss/train", loss, epoch)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            print ("Epoch No. ", epoch, "Loss " , sum(batch_loss)/len(batch_loss))
        
        self.writer.flush()
        F_kfac = FIM(model=net,
                          loader=self.ldr_train,
                          representation=PMatKFAC,
                          device='cuda',
                          n_output=n_c)
        
        F_diag = FIM(model=net,
                          loader=self.ldr_train,
                          representation=PMatDiag,
                          device='cuda',
                          n_output=n_c)

        F_diag = F_diag.get_diag()
        vec_curr = parameters_to_vector(net.parameters())            
        return vec_curr, net, F_kfac, F_diag

    def compute_hessian(self, net, vec):
        with torch.no_grad():
            vector_to_parameters(vec, net.parameters())
            return net.state_dict()

    def train_ivon(self, net, i, n_c):
        net.train()
        optimizer =  ivon.IVON(net.parameters(), 
                               lr=self.args['eta'],
                               ess=round(self.dataset_size, -3)+1000,  # round to the nearest 1000
                               weight_decay=self.args['weight_decay'], 
                               hess_init=self.args['hess_init']
                               )

        for epoch in range(self.args['local_epochs']):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args['device']), labels.to(self.args['device'])
                if(self.args['augmentation']==True):
                    images = self.transform_train(images)
                optimizer.zero_grad()
                
                for _ in range(self.args['mc_train']):
                    with optimizer.sampled_params(train=True):
                        log_probs = net(images)
                        loss = self.loss_func(log_probs, labels)
                        self.writer.add_scalar("Client " + str(i) + " Loss/train", loss, epoch)
                        loss.backward()
                        batch_loss.append(loss.item())              
                optimizer.step()

            print ("Epoch No. ", epoch, "Loss " , sum(batch_loss)/len(batch_loss))
            
        self.writer.flush()
        # crate an  exact copy of the network
        hess_copy = copy.deepcopy(net)
        vec_curr = parameters_to_vector(net.parameters()) 
        hess = self.compute_hessian(hess_copy, optimizer.state_dict()["param_groups"][0]['hess'].detach().clone())
        hess_vec = parameters_to_vector(hess.values())
        return vec_curr, net, hess, hess_vec, optimizer
        