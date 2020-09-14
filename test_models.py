#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 17:51:38 2020

@author: fa19
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def my_get_neighs_order(order_path):
    adj_mat_order = np.load(order_path)
    neigh_orders = np.zeros((len(adj_mat_order), 7))
    neigh_orders[:,0:6] = adj_mat_order-1
    neigh_orders[:,6] = np.arange(len(adj_mat_order))
    neigh_orders = np.ravel(neigh_orders).astype(np.int64)
    
    return neigh_orders


class Linear_simple(nn.Module):
    def __init__(self,in_ch, out_ch):
        super(Linear_simple, self).__init__()
        
        self.fc1 = nn.Linear(in_ch, out_ch)
        
    def forward(self, x):
        
        out = self.fc1(x)
        
        return out
    
    
class Convolutional_simple(nn.Module):
    def __init__(self,in_ch, out_ch):
        super(Convolutional_simple, self).__init__()
        
        self.conv1 = nn.Conv1d(in_ch, 64, kernel_size = 1, stride = 1)
        
        self.conv2 = nn.Conv1d(64, 64, kernel_size=9, stride = 9)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size = 1, stride = 1)
        
        self.af = nn.ReLU()
        
        self.fc1 = nn.Linear(128*4551, out_ch)
        
    def forward(self, x):
        
        x2 = self.conv1(x)
        x2 = self.af(x2)
        x3 = self.conv2(x2)
        x3 = self.af(x3)

        x4 = self.conv3(x3)
        x4 = self.af(x4)
        x5 = x4.reshape(x4.size(0), -1)
        
        out = self.fc1(x5)
        return out


class Simple_Convolution_with_dropout(nn.Module):
    def __init__(self,in_ch, out_ch):
        super(Simple_Convolution_with_dropout, self).__init__()
        
        self.conv1 = nn.Conv1d(in_ch, 64, kernel_size = 1, stride = 1)
        
        self.conv2 = nn.Conv1d(64, 64, kernel_size=9, stride = 9)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size = 1, stride = 1)
        
        self.af = nn.ReLU()
        
        self.fc1 = nn.Linear(128*4551, out_ch)
        self.dropout = nn.Dropout()
        
    def forward(self, x):
        
        x2 = self.conv1(x)
        x2 = self.af(x2)
        x3 = self.conv2(x2)
        x3 = self.af(x3)

        x4 = self.conv3(x3)
        x4 = self.af(x4)        
        x4 = self.dropout(x4)

        x5 = x4.reshape(x4.size(0), -1)
        out = self.fc1(x5)
        return out
    
    
class Conv_with_nebs(nn.Module):
    def __init__(self,in_ch, out_ch):
        super(Conv_with_nebs, self).__init__()
        self.nebs = my_get_neighs_order('/home/fa19/Documents/my_version_spherical_unet/neighbour_indices/adj_mat_order_6.npy')
        
        self.conv1 = nn.Conv1d(in_ch, 64, kernel_size = 1, stride = 1)
        
        self.conv2 = nn.Conv1d(64, 64, kernel_size=7, stride = 7)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size = 1, stride = 1)
        
        self.af = nn.ReLU()
        
        self.fc1 = nn.Linear(128*10242, out_ch)
        self.dropout = nn.Dropout()
        
        self.mpool = nn.MaxPool1d(kernel_size=7, stride = 7)
        
        
    def forward(self, x):
        
        x2 = self.conv1(x)
        x2 = self.af(x2)

        x_nebs = x2[:,:,self.nebs].squeeze(2)

        x3 = self.conv2(x_nebs)
        x3 = self.af(x3)
        x3 = x3[:,:,self.nebs[:10242*7]].squeeze(2)
        x3 = self.mpool(x3)

        x4 = self.conv3(x3)
        x4 = self.af(x4)        
        x4 = self.dropout(x4)

        x5 = x4.reshape(x4.size(0), -1)
        out = self.fc1(x5)
        return out   


class FC_one(nn.Module):
    def __init__(self,in_ch, out_ch):
        super(FC_one, self).__init__()
        self.nebs = my_get_neighs_order('/home/fa19/Documents/Spherical-UNet/neighbour_indices/adj_mat_order_6.npy')
        
        self.conv1 = nn.Conv1d(in_ch, 64, kernel_size = 1, stride = 1)
        
        self.conv2 = nn.Conv1d(64, 64, kernel_size=7, stride = 7)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size = 1, stride = 1)
        
        self.af = nn.ReLU()
        
        self.fc1 = nn.Linear(128*10242, out_ch)
        self.dropout = nn.Dropout()
        
        self.mpool = nn.MaxPool1d(kernel_size=7, stride = 7)
        
        
    def forward(self, x):
        
        x2 = self.conv1(x)
        x2 = self.af(x2)

        x_nebs = x2[:,:,self.nebs[:10242*7]].squeeze(2)

        x3 = self.conv2(x_nebs)
        x3 = self.af(x3)
      
        x4 = self.conv3(x3)
        x4 = self.af(x4)        
        x4 = self.dropout(x4)

        x5 = x4.reshape(x4.size(0), -1)
        out = self.fc1(x5)
        return out   
    
    
