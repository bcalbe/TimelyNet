import argparse
import os
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import sys
sys.path.append('./')
import numpy as np

from TCP.model import TCP
from TCP.data import CARLA_Data
from TCP.config import GlobalConfig
import json
import copy

from TimelyNet.INN import load_model,hyperparameter_generation
from TimelyNet.Lookup import Lookuptable
from time import time
import os

import matplotlib.pyplot as plt



os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_TCP(config = None, resume_path = None):
    model = TCP(config,backbone_name = "Superres")
    #-----------------------------------------------------------------------------------
    #loading a model trained on multi gpu to single gpu will cause key mistach error
    #Specifically, the state_dict of the ckpt model will have "model" prefix in the keys
    #-----------------------------------------------------------------------------------
    if resume_path is not None:
        init = torch.load(resume_path,map_location="cuda:0")
        pre_trained_dict = {}
        for k,v in init['state_dict'].items():
            pre_trained_dict[k.replace("model.", "")] = v
        # pre_trained_dict.pop("join_traj.0.weight")
        # pre_trained_dict.pop("speed_branch.0.weight")
        model.load_state_dict(pre_trained_dict, strict = False)
    for param in model.parameters():
        param.requires_grad = False
    return model


def fusion_INN_lookup(arch_zoo_INN, arch_zoo_lookup,model):
    arch_zoo = []
    test_model = copy.deepcopy(model)
    for i in range(len(arch_zoo_INN)):
        test_model.perception.set_active_subnet(**arch_zoo_INN[i])
        subnet_INN = test_model.perception.get_active_subnet()
        num_INN  = sum(p.numel() for p in subnet_INN.parameters())   
        test_model.perception.set_active_subnet(**arch_zoo_lookup[i])
        subnet_lookup = test_model.perception.get_active_subnet()
        num_lookup  = sum(p.numel() for p in subnet_lookup.parameters())  
        if num_INN < num_lookup:
            arch_zoo.append(arch_zoo_INN[i])
        else:
            arch_zoo.append(arch_zoo_lookup[i])
    return arch_zoo


def run_model(model, data_loader,desired_latency=None):
    device = torch.device("cuda:0")
    batch = next(iter(data_loader))
    model.to(device)
    model.eval()

    front_img = batch['front_img'].to(device)
    speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
    gt_velocity = batch['speed'].to(dtype=torch.float32)
    gt_velocity = gt_velocity.to(device)
    target_point = batch['target_point'].to(dtype=torch.float32)
    command = batch['target_command']
    gt_waypoints = batch['waypoints'][:,:4,:].to(device)
		
    state = torch.cat([speed, target_point, command], 1).to(device)
    value = batch['value'].view(-1,1)
    feature = batch['feature']
    target_point = target_point.to(device)
    runtime_latency = []
    for i in range(11):
        start_time = time()
        pred = model(front_img, state, target_point)
        end_time = time()
        runtime_latency.append((end_time - start_time)*1000)
    print("runtime latency:", np.mean(runtime_latency[1:]))
    return 0

def latency_test(data_loader=None, model=None, INN_model=None, lookup_table=None):
    arch_zoo = []
    #set the required latency
    ddls = [40,50,60,70,80,90]
    for i in range(len(ddls)):
        d = ddls[i]
        
        if d >=85:
            model.perception.set_max_net()
            arch_zoo.append({"d":[2,2,2,2,2],"e":[0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35], "w":[1,1,1,1,1]})
        else:
            can_INN = hyperparameter_generation(INN_model,d)
            can_lookup = lookup_table.look_up(d)
            can = fusion_INN_lookup(can_INN,can_lookup,model)[0]
            arch_zoo.append(can)
            model.perception.set_active_subnet(**can)
        run_model(model, data_loader)
    return arch_zoo


def control_quality_test(model, data_loader,arch_zoo):
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    
    for i in range(len(arch_zoo)):
        can = arch_zoo[i]
        print("Testing subnet:", can)
        control_quality = []
        throttle_delta_list = []
        steer_delta_list = []
        brake_delta_list = []

        # Calculate the control quality of the current subnet
        for batch in data_loader:
            front_img = batch['front_img'].to(device)
            speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
            gt_velocity = batch['speed'].to(dtype=torch.float32)
            gt_velocity = gt_velocity.to(device)
            target_point = batch['target_point'].to(dtype=torch.float32)
            command = batch['target_command']
            gt_waypoints = batch['waypoints'][:,:4,:].to(device)

            state = torch.cat([speed, target_point, command], 1).to(device)
            value = batch['value'].view(-1,1)
            feature = batch['feature']
            target_point = target_point.to(device)

            model.perception.set_active_subnet(**can)
            pred = model(front_img, state, target_point)

            throttle, steer, brake = model.get_action(pred['mu_branches'], pred['sigma_branches'])

            #actions = get_action(model,pred,command,gt_velocity,target_point,batch_size)

            model.perception.set_max_net()
            pred_max = model(front_img, state, target_point)

            throttle_max, steer_max, brake_max = model.get_action(pred_max['mu_branches'], pred_max['sigma_branches'])

            throttle_delta = torch.abs(throttle - throttle_max)
            steer_delta = torch.abs(steer - steer_max)
            brake_delta = torch.abs(brake - brake_max)


            throttle_delta_list.append(throttle_delta.cpu().detach().numpy())
            steer_delta_list.append(steer_delta.cpu().detach().numpy())
            brake_delta_list.append(brake_delta.cpu().detach().numpy())

        #get the maximum difference
        max_throttle_delta = np.max(throttle_delta_list)
        max_steer_delta = np.max(steer_delta_list)
        max_brake_delta = np.max(brake_delta_list)


        if max_throttle_delta == 0 and max_steer_delta == 0 and max_brake_delta == 0:
            control_quality = 1.0
        else:
            control_quality = 1- (np.array(throttle_delta_list)/ max_throttle_delta + np.array(steer_delta_list)/max_steer_delta + np.array(brake_delta_list)/max_brake_delta)/3.0
        print("Control quality for subnet: {:.2f}".format(np.mean(control_quality)))

    return 0

if __name__=="__main__":
    #loading the model and dataset
    config = GlobalConfig()
    batch_size = 1
    
    resume_path="./checkpoint/TCP_model/res50_ta_4wp_123+10ep.ckpt"   
    model = load_TCP(config=config, resume_path=resume_path) 
    INN_model = load_model(72) 
    lookup_table = Lookuptable()


    data_set = CARLA_Data(root=config.root_dir_all, data_folders=config.train_data, img_aug = config.img_aug)
    print(len(data_set))
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=1)

    arch_zoo = latency_test(data_loader=data_loader, model=model, INN_model=INN_model, lookup_table=lookup_table)

    control_quality_test(model = model, data_loader=data_loader, arch_zoo=arch_zoo)
