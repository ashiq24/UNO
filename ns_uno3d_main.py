# Codes for section: Results on Navier Stocks Equation (3D)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from data_load_navier_stocks import *
import matplotlib.pyplot as plt
from navier_stokes_uno3d import Uno3D_T40, Uno3D_T20, Uno3D_T10, Uno3D_T9
import operator
import random
from functools import reduce
from functools import partial
from ns_train_3d import train_model_3d
from timeit import default_timer
from utilities3 import *
from Adam import Adam
from torchsummary import summary

# for conda: from torchinfo import summary
import gc
import math

torch.cuda.set_device(2)
random.seed(10102)

# Hyper parameters
# weight deacy = 1e-5
# learning rate 0.003


S = 64  # resolution SxS
T_in = 10  # input time interval (0 - T_in)
T_f = 10  # output time interval (T_in -  = T_in+T_f)
ntrain = 9000  # number of training instances
ntest = 1000  # number of test instances
nval = 1000  # number of validation instances
batch_size = 16
width = 8  # Uplifting dimesion
inwidth = 6  # dimension of UNO input ( a(x,y,t), + positional embedding )
epochs = 500

# Following code load data from two separate files containing Navier-Stokes equation simulation

train_a_1, train_u_1, test_a_1, test_u_1 = load_NS_(
    "path to navier stokes simulation with viscosity 1e-5 with 1000 instances",
    4000,
    2000,
    Sample_num=6000,
    T_in=T_in,
    T=T_f,
    size=S,
)
train_a_2, train_u_2, test_a_2, test_u_2 = load_NS_(
    "path to navier stokes simulation with viscosity 1e-5 with 5000 instances",
    4000,
    1000,
    Sample_num=5000,
    T_in=T_in,
    T=T_f,
    size=S,
)

a = torch.cat([train_a_1, train_a_2, test_a_1, test_a_2], dim=0)
u = torch.cat([train_u_1, train_u_2, test_u_1, test_u_2], dim=0)
indexs = [i for i in range(a.shape[0])]
random.shuffle(indexs)
train_a, val_a, test_a = (
    a[indexs[:ntrain]],
    a[indexs[ntrain : ntrain + nval]],
    a[indexs[ntrain + nval :]],
)
train_u, val_u, test_u = (
    u[indexs[:ntrain]],
    u[indexs[ntrain : ntrain + nval]],
    u[indexs[ntrain + nval :]],
)
train_a = train_a.reshape(ntrain, S, S, T_in)
val_a = val_a.reshape(nval, S, S, T_in)
test_a = test_a.reshape(ntest, S, S, T_in)


train_a = train_a.reshape(ntrain, S, S, T_in, 1)
val_a = val_a.reshape(nval, S, S, T_in, 1)
test_a = test_a.reshape(ntest, S, S, T_in, 1)
gc.collect()


train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_a, train_u),
    batch_size=batch_size,
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(val_a, val_u), batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False
)
# any 3d models can be trained vai train_model_3d function

# model = Uno3D_T10(inwidth,width,pad = 3,factor = 1).cuda()
model = Uno3D_T40(inwidth, width, pad=3, factor=1).cuda()
summary(model, (64, 64, 6, 1))

train_model_3d(
    model,
    train_loader,
    val_loader,
    test_loader,
    ntrain,
    nval,
    ntest,
    weight_path="UNO3D.pt",
    T_f=T_f,
    batch_size=batch_size,
    epochs=epochs,
    learning_rate=0.003,
    scheduler_step=100,
    scheduler_gamma=0.5,
    weight_decay=1e-5,
)
