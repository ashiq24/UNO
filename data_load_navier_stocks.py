import torch.utils.data as data
import torchvision
import gc
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
from utilities3 import MatReader
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchvision import transforms as T
from torchvision import models, datasets
import os
from random import randint
import urllib
import zipfile


def load_NS_(path1,train,test,Sample_num = 1000, batch = 20,T_in = 10,T= 10,size=64):
    """
    train, test = number of train test split
    Sample_num = train + test
    batch = Must be equal to the batch size data generation, Dafault is set to 20 in both places
    T_in = [0,T_in] initial interval
    T = (t_in, T_in+T] later interva;
    size x size = simulation resolution
    """
    reader = MatReader(path1)
    train_a = None
    train_u = None
    test_a = None
    test_u = None
    idx = 0
    for i in range(Sample_num//batch):
        idx+=batch
        k_a = reader.read_field('u'+str(i))[:,:,:,:T_in]
        k_u = reader.read_field('u'+str(i))[:,:,:,T_in:T_in+T]
        k_a = torch.nn.functional.interpolate(k_a.permute(0,3,1,2),size=(size,size),mode = 'bilinear',align_corners=True).permute(0,2,3,1)
        k_u = torch.nn.functional.interpolate(k_u.permute(0,3,1,2),size=(size,size),mode = 'bilinear',align_corners=True).permute(0,2,3,1)
        if idx <= train:
            if train_a is None:
                train_a = k_a
                train_u = k_u
            else:
                train_a = torch.cat([train_a,k_a], dim = 0)
                train_u = torch.cat([train_u,k_u],dim = 0)
        else:
            if test_a is None:
                test_a = k_a
                test_u = k_u
            else:
                test_a = torch.cat([test_a,k_a], dim = 0)
                test_u = torch.cat([test_u,k_u], dim = 0)
    
    print("train shape",train_a.shape, train_u.shape)
    print("test shape", test_a.shape, test_u.shape)
    return train_a,train_u, test_a, test_u

   
