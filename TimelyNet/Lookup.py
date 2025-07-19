
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import json
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "2"



class Lookuptable(object):
    def __init__(self,arch_path="data/arch.npy",latency_path="data/latency.npy"):

        self.latency = np.load(latency_path)
        self.archs  = np.load(arch_path)


        

    def one_hot_encode(self,arch_config=None):

        d = [0,1,2]
        e = [0.2,0.25,0.35]
        w = [0,1,2]
        features = np.array([])

        total_dim = len(d)*5 + len(e)*18
        if arch_config is None:
            archs = self.archs

        for arch in archs:
            d_list = arch["d"]
            d_onehot = np.eye(3)[d_list].reshape(-1)
            e_list = arch["e"]
            e_label = [np.where(np.array(e)==i)[0][0] for i in e_list]
            e_onehot = np.eye(3)[e_label].reshape(-1)
            feature = np.concatenate([d_onehot,e_onehot])
            features = np.concatenate([features,feature])
        return features.reshape(-1,total_dim)

    def one_hot_decode(self,features=None):
        d = [0,1,2]
        e = [0.2,0.25,0.35]
        w = [0,1,2]
        archs = []

        if features is None:
            features = self.data
        d_dim = 5*3
        e_dim = 18*3
        for feature in features:
            d_onehot = feature[:d_dim]
            d_list = np.argmax(d_onehot.reshape(5,3),axis=1)
            e_onehot = feature[d_dim:d_dim+e_dim]
            e_class = np.argmax(e_onehot.reshape(18,3),axis=1)
            e_list = [e[i] for i in e_class]
            arch = {"d":d_list,"e":e_list,"w":[1,1,1,1,1]}
            archs.append(arch)
        return archs

    def look_up(self,d):
        d = np.round(d)
        if d > np.max(self.latency):
            d = np.max(self.latency)
        elif d < np.min(self.latency):
            d = np.min(self.latency)
        arch = self.archs[self.latency==d]
        arch = self.one_hot_decode(arch)
        return arch
    
    
    

if __name__ == "__main__":
    #L = Latency_dataset_E2E()
    L = Lookuptable()
    for i in range(10):
        time1 = time.time()
        L.look_up(50)
        time2 = time.time()
        print(time2-time1)
