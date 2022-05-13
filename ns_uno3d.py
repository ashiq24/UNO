
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from data_ns import *
import matplotlib.pyplot as plt
from UNO3D import Uno3D_T40,Uno3D_T20,Uno3D_T10,Uno3D_T9
import operator
import random
from functools import reduce
from functools import partial
from train_3d import train_model_3d
from timeit import default_timer
from utilities3 import *
from Adam import Adam
from torchsummary import summary
import gc
import math
plt.rcParams['figure.figsize'] = [6, 30]
plt.rcParams['image.interpolation'] = 'nearest'


S = 64
T_in = 10
T_f = 40 
step = 1
ntrain = 1720
ntest = 480
nval = 200
batch_size = 16
width = 8
inwidth = 4
epochs = 700
train_a_1, train_u_1, test_a_1, test_u_1 = load_NS_("/home/ashiq/Desktop/Neural Operator codes/NS_data/ns_data_1200_T50_v001_dt0001.mat"\
                                                    ,1000,200,Sample_num = 1200,T_in=T_in, T = T_f, size = 64)
train_a_2, train_u_2, test_a_2, test_u_2 = load_NS_("/home/ashiq/Desktop/Neural Operator codes/NS_data/ns_data_2_1200_T50_v001_dt0001.mat"\
                                                    ,1000 ,200,Sample_num = 1200,T_in=T_in, T = T_f, size = 64)
a = torch.cat([train_a_1,train_a_2,test_a_1,test_a_2], dim = 0)
u = torch.cat([train_u_1,train_u_2,test_u_1,test_u_2],dim = 0)
indexs = [i for i in range(a.shape[0])]
random.shuffle(indexs)
train_a,val_a,test_a = a[indexs[:ntrain]], a[indexs[ntrain:ntrain+nval]],a[indexs[ntrain+nval:]]
train_u,val_u,test_u = u[indexs[:ntrain]],u[indexs[ntrain:ntrain+nval]],u[indexs[ntrain+nval:]]
train_a = train_a.reshape(ntrain,S,S,T_in)
val_a = val_a.reshape(nval,S,S,T_in)
test_a = test_a.reshape(ntest,S,S,T_in)



train_a = train_a.reshape(ntrain,S,S,T_in,1)
val_a = val_a.reshape(nval,S,S,T_in,1)
test_a = test_a.reshape(ntest,S,S,T_in,1)
gc.collect()


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u),\
                                           batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_a, val_u),\
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), \
                                          batch_size=batch_size, shuffle=False)

model = Uno3D_T40(inwidth,width,pad = 1,factor = 1).cuda()
summary(model, (64, 64,6,1))
train_model_3d(model,train_loader,val_loader,test_loader, ntrain,nval,ntest,\
            weight_path = 'UNO3D(New)-10e3.pt',T_f=T_f,batch_size=batch_size,\
               epochs=epochs,learning_rate= 0.0008,\
            x_normalizer = None, y_normalizer = None,scheduler_step= 100,\
               scheduler_gamma= 0.7,weight_dec = 1e-3)