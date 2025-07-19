import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import FrEIA.framework as Ff
import FrEIA.modules as Fm
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom
import sys
import numpy as np

def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in,512), nn.ReLU(),
                        #  nn.Linear(512, 1024), nn.ReLU(),
                        #  nn.Linear(1024,512), nn.ReLU(),
                         nn.Linear(512,  c_out))

def generate_model(ndim_tot):
    nodes = [InputNode(ndim_tot, name='input')]

    for k in range(4):
        nodes.append(Node(nodes[-1],
                      GLOWCouplingBlock,
                      {'subnet_constructor':subnet_fc, 'clamp':2.0},
                      name=F'coupling_{k}'))
        nodes.append(Node(nodes[-1],
                      PermuteRandom,
                      {'seed':k},
                      name=F'permute_{k}'))

    nodes.append(OutputNode(nodes[-1], name='output'))

    model = ReversibleGraphNet(nodes, verbose=False)
    return model

def load(model,name):
    state_dicts = torch.load(name)
    model.load_state_dict(state_dicts['net'])
    return model


def load_model(ndim_tot):
    model = generate_model(ndim_tot)
    model = load(model,'./checkpoint/INN/inn_e2e_optimal_512_4.pt')
    model.eval()
    return model


def hyperparameter_generation(model,d, a = None):
    #--------------------------------------------------------------
    #Successfully parameters inn_e2e_acc
    # ndim_tot = 72
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = model.to(device)
    # ndim_x = 69
    # a = 60
    # d = np.round(d*1000)-70
    #----------------------------------------------------------------
    ndim_tot = 72
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    ndim_x = 69
    d_offset =10
    d = d-d_offset
    #zy = torch.tensor([0.5]*(ndim_tot-2)+[a,d]).to(torch.float32)
    zy = torch.tensor([0.0]*(ndim_tot-1)+[d]).to(torch.float32)
    zy = zy.unsqueeze(0).to(device)
    x_revese,log_jac = model(zy,rev=True)
    feature = x_revese[:, :ndim_x]
    arch = one_hot_decode(feature)
    # for i in range(10):
    #     forward_validation(feature,INN_model=model,device=device,latency = d,accuracy = a)
    return arch

def forward_validation(feature,INN_model = None,device = 'cuda', latency=None, accuracy=None):
    x = torch.cat([feature,torch.zeros(1,3).to("cuda")],axis = 1)
    zy, log_jac  = INN_model(x)
    l = zy[:, -1]
    a = zy[:, -2]
    latency_err = latency-l
    accuracy_err = accuracy-a
    return latency_err,accuracy_err

def one_hot_decode(features=None):
    d = [0,1,2]
    e = [0.2,0.25,0.35]
    w = [0,1,2]
    archs = []

    # if features is None:
    #     features = self.data
    d_dim = 5*3
    e_dim = 18*3
    for feature in features:
        feature = feature.cpu().detach().numpy()
        d_onehot = feature[:d_dim]
        d_list = np.argmax(d_onehot.reshape(5,3),axis=1)
        e_onehot = feature[d_dim:d_dim+e_dim]
        e_class = np.argmax(e_onehot.reshape(18,3),axis=1)
        e_list = [e[i] for i in e_class]
        arch = {"d":d_list,"e":e_list,"w":[1,1,1,1,1]}
        archs.append(arch)
    return archs



if __name__ == '__main__':
    ndim_tot = 72
    model = load_model(ndim_tot)
    hyperparameter_generation(model,d)
    