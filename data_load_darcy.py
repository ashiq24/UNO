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



def load_data_darcy(r,ntrain,ntest,TRAIN_PATH):
    """
    r = Subsampling rate
    ntrain, ntest = will return the data spliting into two with ntrain and ntest samples 
    """
    h = int(((421 - 1)/r) + 1)
    s = h
    print("resolution S ",s)
    reader = MatReader(TRAIN_PATH)
    x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s]
    y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]

    reader.load_file(TRAIN_PATH)
    x_test = reader.read_field('coeff')[-ntest:,::r,::r][:,:s,:s]
    y_test = reader.read_field('sol')[-ntest:,::r,::r][:,:s,:s]

    x_train = x_train.reshape(ntrain,s,s,1)
    x_test = x_test.reshape(ntest,s,s,1)

    return x_train, y_train, x_test,y_test

    
