import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from data_load_darcy import *
import random
import matplotlib.pyplot as plt
from darcy_flow_uno2d import UNO,UNO_P_13,UNO_P_9
import operator
from functools import reduce
from functools import partial
from train_darcy import train_model
from timeit import default_timer
from utilities3 import *
from Adam import Adam
from torchsummary import summary
import gc
import math
plt.rcParams['figure.figsize'] = [6, 30]
plt.rcParams['image.interpolation'] = 'nearest'
torch.manual_seed(10001)
random.seed(10001)

import sys
import logging


train_a_1, train_u_1, test_a_1, test_u_1 = load_data_darcy(2,800,200,"Path to data file1")
train_a_2, train_u_2, test_a_2, test_u_2 = load_data_darcy(2,800,200,"Path to data file2")

sub = 2 # subsampling rate 
S = 211 # Grid size/ resolution
# single input and output
T_in = 1
T_f = 1 
# number of train, test and validation samples
ntrain = 1400
nval = 200
ntest = 400
batch_size = 16
width = 32 #
inwidth = 3
epochs = 700
a = torch.cat([train_a_1,train_a_2,test_a_1,test_a_2], dim = 0)
u = torch.cat([train_u_1,train_u_2,test_u_1,test_u_2],dim = 0)

indexs = [i for i in range(a.shape[0])]
random.shuffle(indexs)

train_a,val_a,test_a = a[indexs[:ntrain]], a[indexs[ntrain:ntrain+nval]],a[indexs[ntrain+nval:]]
train_u,val_u,test_u = u[indexs[:ntrain]],u[indexs[ntrain:ntrain+nval]],u[indexs[ntrain+nval:]]
print(train_a.shape,val_a.shape,test_a.shape)
train_a = train_a.reshape(ntrain,S,S,T_in)
val_a = val_a.reshape(nval,S,S,T_in)
test_a = test_a.reshape(ntest,S,S,T_in)

gc.collect()



train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_a, val_u), batch_size=batch_size, shuffle=True)

model = UNO(inwidth,width,pad = 8).cuda()
#model = UNO_P_13(inwidth,width,pad = 8).cuda()
summary(model, (S, S,1))
gc.collect()
train_model(model,train_loader,val_loader,test_loader, ntrain,nval,ntest,S,'Darcy-D13-421.pt',\
            T_f=T_f,batch_size=batch_size,epochs=epochs,learning_rate= 0.001,\
                scheduler_step= 100,scheduler_gamma= 0.7,weight_decay = 1e-3)
