import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from data_load_navier_stocks import *
import matplotlib.pyplot as plt
from navier_stokes_uno2d import UNO, UNO_P
import operator
import random
from functools import reduce
from functools import partial
from ns_train_2d import train_model
from timeit import default_timer
from utilities3 import *
from Adam import Adam
from torchsummary import summary
import gc
import math


S = 64 # resolution SxS
T_in = 10 # input time interval (0 - T_in)
T_f = 40 # output time interval (T_in -  = T_in+T_f)
ntrain = 1720 # number of training instances 
ntest = 480 # number of test instances 
nval = 200 # number of validation instances 
batch_size = 20
width = 32 # Uplifting dimesion
inwidth = 12 # dimension of UNO input ( 10 time step + (x,y) location )
epochs = 700
# Following code load data from two separate files containing Navier-Stokes equation simulation
train_a_1, train_u_1, test_a_1, test_u_1 = load_NS_("path to navier stokes simulation with viscosity 1e-3 with 1200 instances"\
                                                    ,1000,200,Sample_num = 1200,T_in=T_in, T = T_f, size = S)
train_a_2, train_u_2, test_a_2, test_u_2 = load_NS_("path to navier stokes simulation with viscosity 1e-3 with 1200 instances"\
                                                    ,1000 ,200,Sample_num = 1200,T_in=T_in, T = T_f, size=S)
a = torch.cat([train_a_1,train_a_2,test_a_1,test_a_2], dim = 0)
u = torch.cat([train_u_1,train_u_2,test_u_1,test_u_2],dim = 0)
indexs = [i for i in range(a.shape[0])]
random.shuffle(indexs)
train_a,val_a,test_a = a[indexs[:ntrain]], a[indexs[ntrain:ntrain+nval]],a[indexs[ntrain+nval:]]
train_u,val_u,test_u = u[indexs[:ntrain]],u[indexs[ntrain:ntrain+nval]],u[indexs[ntrain+nval:]]
train_a = train_a.reshape(ntrain,S,S,T_in)
val_a = val_a.reshape(nval,S,S,T_in)
test_a = test_a.reshape(ntest,S,S,T_in)

gc.collect()


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u),\
                                           batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_a, val_u),\
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), \
                                          batch_size=batch_size, shuffle=False)

model = UNO(inwidth,width)
summary(model, (64, 64,10))
train_model(model,train_loader,val_loader,test_loader, ntrain,nval,ntest,\
            weight_path = 'UNO-10e3.pt',T_f=T_f,batch_size=batch_size,epochs=epochs,learning_rate= 0.0008,scheduler_step= 100,scheduler_gamma= 0.7,weight_decay = 1e-3)
