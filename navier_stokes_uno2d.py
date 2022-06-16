# Codes for section: Results on Navier Stocks Equation (2D)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import matplotlib.pyplot as plt
from integral_operators import *
import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *
from Adam import Adam


torch.manual_seed(0)
np.random.seed(0)

class UNO_P(nn.Module):
    def __init__(self,in_width, width,pad = 0, factor = 1):
        super(UNO_P, self).__init__()

        """
        The overall network. It contains 7 integral operator.
        1. Lift the input to the desire channel dimension by  self.fc, self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .


        input: the solution of the first 10 timesteps (u(1), ..., u(10)).
        input shape: (batchsize, x=S, y=S, t=10)
        output: the solution of the next timesteps
        output shape: (batchsize, x=S, y=S, t=1)
        Here SxS is the spatial resolution

        in_width = 12 (10 input time steps + (x,y) location)
        with = uplifting dimension
        pad = padding the domian for non-periodic input
        factor = factor for scaling up/down the co-domain dimension at each integral operator

        """
        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  # pad the domain if input is non-periodic

        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)

        self.L0 = OperatorBlock_2D(self.width, 2*factor*self.width,32, 32, 14, 14)

        self.L1 = OperatorBlock_2D(2*factor*self.width, 4*factor*self.width, 16, 16, 6,6)

        self.L2 = OperatorBlock_2D(4*factor*self.width, 8*factor*self.width, 8, 8,3,3)
        
        self.L3 = OperatorBlock_2D(8*factor*self.width, 8*factor*self.width, 8, 8,3,3)
        
        self.L4 = OperatorBlock_2D(8*factor*self.width, 4*factor*self.width, 16, 16,3,3)

        self.L5 = OperatorBlock_2D(8*factor*self.width, 2*factor*self.width, 32, 32,6,6)

        self.L6 = OperatorBlock_2D(4*factor*self.width, self.width, 64, 64,14,14) # will be reshaped


        self.fc1 = nn.Linear(2*self.width, 3*self.width)
        self.fc2 = nn.Linear(3*self.width + self.width//2, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x_fc = self.fc(x)
        x_fc = F.gelu(x_fc)

        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        x_fc0 = F.pad(x_fc0, [self.padding,self.padding, self.padding,self.padding])
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.L0(x_fc0,D1//2,D2//2)

        x_c1 = self.L1(x_c0,D1//4,D2//4)


        x_c2 = self.L2(x_c1,D1//8,D2//8)

        
        x_c3 = self.L3(x_c2,D1//8,D2//8)


        x_c4 = self.L4(x_c3 ,D1//4,D2//4)
        x_c4 = torch.cat([x_c4, x_c1], dim=1)

        x_c5 = self.L5(x_c4 ,D1//2,D2//2)
        x_c5 = torch.cat([x_c5, x_c0], dim=1)

        x_c6 = self.L6(x_c5,D1,D2)
        x_c6 = torch.cat([x_c6, x_fc0], dim=1)

        if self.padding!=0:
            x_c6 = x_c6[..., self.padding:-self.padding, self.padding:-self.padding]

        x_c6 = x_c6.permute(0, 2, 3, 1)

        x_fc1 = self.fc1(x_c6)
        x_fc1 = F.gelu(x_fc1)

        x_fc1 = torch.cat([x_fc1, x_fc], dim=3)
        x_out = self.fc2(x_fc1)
        
        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

#####
# The following models UNO has same sets of parameter 
# it has different scaling factors for domains and co-domains.
# ####    
class UNO(nn.Module):
    def __init__(self,in_width, width,pad = 0, factor = 3/4):
        super(UNO, self).__init__()


        self.in_width = in_width # input channel
        self.width = width 
        self.factor = factor
        self.padding = pad  

        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)

        self.L0 = OperatorBlock_2D(self.width, 2*factor*self.width,48, 48, 18, 18)

        self.L1 = OperatorBlock_2D(2*factor*self.width, 4*factor*self.width, 32, 32, 14,14)

        self.L2 = OperatorBlock_2D(4*factor*self.width, 8*factor*self.width, 16, 16,6,6)
        
        self.L3 = OperatorBlock_2D(8*factor*self.width, 8*factor*self.width, 16, 16,6,6)
        
        self.L4 = OperatorBlock_2D(8*factor*self.width, 4*factor*self.width, 32, 32,6,6)

        self.L5 = OperatorBlock_2D(8*factor*self.width, 2*factor*self.width, 48, 48,14,14)

        self.L6 = OperatorBlock_2D(4*factor*self.width, self.width, 64, 64,18,18) # will be reshaped

        self.fc1 = nn.Linear(2*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
         
        x_fc = self.fc(x)
        x_fc = F.gelu(x_fc)

        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        
        
        x_fc0 = F.pad(x_fc0, [self.padding,self.padding, self.padding,self.padding])
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]
        
        x_c0 = self.L0(x_fc0,int(D1*self.factor),int(D2*self.factor))
        x_c1 = self.L1(x_c0 ,D1//2,D2//2)

        x_c2 = self.L2(x_c1 ,D1//4,D2//4)        
        x_c3 = self.L3(x_c2,D1//4,D2//4)
        x_c4 = self.L4(x_c3,D1//2,D2//2)
        x_c4 = torch.cat([x_c4, x_c1], dim=1)
        x_c5 = self.L5(x_c4,int(D1*self.factor),int(D2*self.factor))
        x_c5 = torch.cat([x_c5, x_c0], dim=1)
        x_c6 = self.L6(x_c5,D1,D2)
        x_c6 = torch.cat([x_c6, x_fc0], dim=1)

        if self.padding!=0:
            x_c6 = x_c6[..., :-self.padding, :-self.padding]

        x_c6 = x_c6.permute(0, 2, 3, 1)
        
        x_fc1 = self.fc1(x_c6)
        x_fc1 = F.gelu(x_fc1)
        
        x_out = self.fc2(x_fc1)
        
        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


###
# UNO for high resolution (256x256) navier stocks simulations
###

class UNO_S256(nn.Module):
    def __init__(self, in_width, width,pad = 0, factor = 1):
        super(UNO_S256, self).__init__()
        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  # pad the domain if input is non-periodic

        self.fc = nn.Linear(self.in_width, 16)

        self.fc0 = nn.Linear(16, self.width) # input channel is 3: (a(x, y), x, y)

        self.L0 = OperatorBlock_2D(self.width, 2*factor*self.width,64, 64, 32, 33)

        self.L1 = OperatorBlock_2D(2*factor*self.width, 4*factor*self.width, 16, 16, 8,9)

        self.L2 = OperatorBlock_2D(4*factor*self.width, 8*factor*self.width, 8, 8,4,5)
        
        self.L3 = OperatorBlock_2D(8*factor*self.width, 8*factor*self.width, 8, 8,4,5)
        
        self.L4 = OperatorBlock_2D(8*factor*self.width, 4*factor*self.width, 16, 16,4,5)

        self.L5 = OperatorBlock_2D(8*factor*self.width, 2*factor*self.width, 64, 64,8,9)

        self.L6 = OperatorBlock_2D(4*factor*self.width, self.width, 256, 256,32,32) # will be reshaped

        self.fc1 = nn.Linear(2*self.width, 3*self.width)
        self.fc2 = nn.Linear(3*self.width + 16, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x_fc = self.fc(x)
        x_fc = F.gelu(x_fc)

        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        x_fc0 = F.pad(x_fc0, [self.padding,self.padding, self.padding,self.padding])
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.L0(x_fc0,D1//4,D2//4)
        x_c1 = self.L1(x_c0,D1//16,D2//16)
        x_c2 = self.L2(x_c1,D1//32,D2//32)        
        x_c3 = self.L3(x_c2,D1//32,D2//32)
        x_c4 = self.L4(x_c3 ,D1//16,D2//16)
        x_c4 = torch.cat([x_c4, x_c1], dim=1)
        x_c5 = self.L5(x_c4 ,D1//4,D2//4)
        x_c5 = torch.cat([x_c5, x_c0], dim=1)

        x_c6 = self.L6(x_c5,D1,D2)
        #print(x.shape)
        x_c6 = torch.cat([x_c6, x_fc0], dim=1)

        if self.padding!=0:
            x_c6 = x_c6[..., self.padding:-self.padding, self.padding:-self.padding]

        x_c6 = x_c6.permute(0, 2, 3, 1)

        x_fc1 = self.fc1(x_c6)
        x_fc1 = F.gelu(x_fc1)

        x_fc1 = torch.cat([x_fc1, x_fc], dim=3)
        x_out = self.fc2(x_fc1)
        
        return x_out
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
