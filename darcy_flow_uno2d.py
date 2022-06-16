# Codes for section: Results on Darcy Flow Equation

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

###############
#  UNO+ achitechtures
###############
class UNO_P_13(nn.Module):
    def __init__(self, in_width, width,pad = 8, factor = 1):
        super(UNO_P_13, self).__init__()
        """
        The overall network. It contains 13 integral operator.
        1. Lift the input to the desire channel dimension by  self.fc, self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .


        input: Diffusion coefficient (a)
        input shape: (batchsize, x=S, y=S, 1, 1)
        output: the solution of the next timesteps
        output shape: (batchsize, x=S, y=S, 1, 1)
        Spatial Resulution = SxS

        in_width = 3 ;  [a(x,y),x,y]
        with = Up-lifting dimension
        pad = padding the domian for non-periodic input
        factor = factor for scaling up/down the co-domain dimension at each integral operator
        """

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  

        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) 

        self.conv0 = OperatorBlock_2D(self.width, 2*factor*self.width,40, 40, 20, 20)

        self.conv1 = OperatorBlock_2D(2*factor*self.width, 4*factor*self.width, 20, 20, 10,10)

        self.conv2 = OperatorBlock_2D(4*factor*self.width, 8*factor*self.width, 10, 10,5,5)
        
        self.conv3 = OperatorBlock_2D(8*factor*self.width, 16*factor*self.width, 5, 5,3,3)
        
        self.conv4 = OperatorBlock_2D(16*factor*self.width, 16*factor*self.width, 5, 5,3,3)
        
        self.conv5 = OperatorBlock_2D(16*factor*self.width, 16*factor*self.width, 5, 5,3,3)
        
        self.conv6 = OperatorBlock_2D(16*factor*self.width, 16*factor*self.width, 5, 5,3,3)
        
        self.conv7 = OperatorBlock_2D(16*factor*self.width, 16*factor*self.width, 5, 5,3,3)
        
        self.conv8 = OperatorBlock_2D(16*factor*self.width, 16*factor*self.width, 5, 5,3,3)
        
        self.conv9 = OperatorBlock_2D(16*factor*self.width, 8*factor*self.width, 10, 10,3,3)
        
        self.conv10 = OperatorBlock_2D(16*factor*self.width, 4*factor*self.width, 20, 20,5,5)

        self.conv11 = OperatorBlock_2D(8*factor*self.width, 2*factor*self.width, 40, 40,10,10)

        self.conv12 = OperatorBlock_2D(4*factor*self.width, self.width, 85, 85,20,20) # will be reshaped

        self.fc1 = nn.Linear(1*self.width, 2*self.width)
        self.fc2 = nn.Linear(2*self.width, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x_fc = self.fc(x)
        x_fc = F.gelu(x_fc)

        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        x_fc0 = F.pad(x_fc0, [0,self.padding, 0,self.padding])
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.conv0(x_fc0,D1//2,D2//2)
        #print(x.shape)

        x_c1 = self.conv1(x_c0,D1//4,D2//4)

        x_c2 = self.conv2(x_c1,D1//8,D2//8)
        
        x_c3 = self.conv3(x_c2,D1//16,D2//16)
        
        x_c4 = self.conv4(x_c3,D1//16,D2//16)
        
        x_c5 = self.conv5(x_c4,D1//16,D2//16)
        
        x_c6 = self.conv6(x_c5,D1//16,D2//16)
        
        x_c7 = self.conv7(x_c6,D1//16,D2//16)
        
        x_c8 = self.conv8(x_c7,D1//16,D2//16)
        
        x_c9 = self.conv9(x_c8,D1//8,D2//8)
        x_c9 = torch.cat([x_c9, x_c2], dim=1) 
        
        x_c10 = self.conv10(x_c9 ,D1//4,D2//4)
        x_c10 = torch.cat([x_c10, x_c1], dim=1)

        x_c11 = self.conv11(x_c10 ,D1//2,D2//2)
        x_c11 = torch.cat([x_c11, x_c0], dim=1)

        x_c12 = self.conv12(x_c11,D1,D2)
        if self.padding!=0:
            x_c12 = x_c12[..., :-self.padding, :-self.padding]


        x_c12 = x_c12.permute(0, 2, 3, 1)

        x_fc1 = self.fc1(x_c12)
        x_fc1 = F.gelu(x_fc1)

        #x_fc1 = torch.cat([x_fc1, x_fc_1], dim=3)
        x_out = self.fc2(x_fc1)
        
        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

# For the following classes every parameters acts the same as UNO_P_13.
##UNO+ 9 layers   
class UNO_P_9(nn.Module):
    def __init__(self, in_width, width,pad = 8, factor = 1):
        super(UNO_P_9, self).__init__()

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  # pad the domain if input is non-periodic

        self.fc_n1 = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)

        #self.conv_lin = OperatorBlock_2D(self.width, self.width,85, 85, 24, 24)

        self.conv0 = OperatorBlock_2D(self.width, 2*factor*self.width,40, 40, 20, 20)

        self.conv1 = OperatorBlock_2D(2*factor*self.width, 4*factor*self.width, 20, 20, 10,10)

        self.conv2 = OperatorBlock_2D(4*factor*self.width, 8*factor*self.width, 10, 10,5,5)
        
        self.conv3 = OperatorBlock_2D(8*factor*self.width, 16*factor*self.width, 5, 5,3,3)
        
        self.conv4 = OperatorBlock_2D(16*factor*self.width, 16*factor*self.width, 5, 5,3,3)
        
        self.conv5 = OperatorBlock_2D(16*factor*self.width, 8*factor*self.width, 10, 10,3,3)
        
        self.conv6 = OperatorBlock_2D(16*factor*self.width, 4*factor*self.width, 20, 20,5,5)

        self.conv7 = OperatorBlock_2D(8*factor*self.width, 2*factor*self.width, 40, 40,10,10)

        self.conv8 = OperatorBlock_2D(4*factor*self.width, self.width, 85, 85,20,20) # will be reshaped


        self.fc1 = nn.Linear(1*self.width, 2*self.width)
        self.fc2 = nn.Linear(2*self.width, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x_fc_1 = self.fc_n1(x)
        x_fc_1 = F.gelu(x_fc_1)

        x_fc0 = self.fc0(x_fc_1)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        x_fc0 = F.pad(x_fc0, [0,self.padding, 0,self.padding])
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.conv0(x_fc0,D1//2,D2//2)

        x_c1 = self.conv1(x_c0,D1//4,D2//4)

        x_c2 = self.conv2(x_c1,D1//8,D2//8)
        
        x_c3 = self.conv3(x_c2,D1//16,D2//16)
        
        x_c4 = self.conv4(x_c3,D1//16,D2//16)
        x_c4 = F.gelu(x_c4)
 
        x_c5 = self.conv5(x_c4,D1//8,D2//8)
        x_c5 = torch.cat([x_c5, x_c2], dim=1) 
        
        x_c6 = self.conv6(x_c5 ,D1//4,D2//4)
        x_c6 = torch.cat([x_c6, x_c1], dim=1)

        x_c7 = self.conv7(x_c6 ,D1//2,D2//2)
        x_c7 = torch.cat([x_c7, x_c0], dim=1)

        x_c8 = self.conv8(x_c7,D1,D2)

        if self.padding!=0:
            x_c8 = x_c8[..., :-self.padding, :-self.padding]


        x_c8 = x_c8.permute(0, 2, 3, 1)

        x_fc1 = self.fc1(x_c8)
        x_fc1 = F.gelu(x_fc1)

        #x_fc1 = torch.cat([x_fc1, x_fc_1], dim=3)
        x_out = self.fc2(x_fc1)
        
        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

#########
# UNO achitecture
##########

class UNO(nn.Module):
    def __init__(self,in_width, width,pad = 8, factor = 3/4):
        super(UNO, self).__init__()
        
        self.in_width = in_width # input channel
        self.width = width 
        self.factor = factor
        self.padding = pad 
        
        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)
        
        self.conv0 = OperatorBlock_2D(self.width, 2*factor*self.width,64, 64, 28, 28)

        self.conv1 = OperatorBlock_2D(2*factor*self.width, 4*factor*self.width, 32, 32, 16,16)

        self.conv2 = OperatorBlock_2D(4*factor*self.width, 8*factor*self.width, 16, 16,8,8)
        
        self.conv3 = OperatorBlock_2D(8*factor*self.width, 8*factor*self.width, 16, 16,8,8)
        
        self.conv4 = OperatorBlock_2D(8*factor*self.width, 4*factor*self.width, 32, 32,8,8)

        self.conv5 = OperatorBlock_2D(8*factor*self.width, 2*factor*self.width, 64, 64,16,16)

        self.conv6 = OperatorBlock_2D(4*factor*self.width, self.width, 85, 85,28,28) # will be reshaped

        self.fc1 = nn.Linear(1*self.width, 2*self.width)
        self.fc2 = nn.Linear(2*self.width, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)


        x_fc = self.fc(x)
        x_fc = F.gelu(x_fc)

        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        
        
        x_fc0 = F.pad(x_fc0, [0,self.padding, 0,self.padding])#
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]
        

        x_c0 = self.conv0(x_fc0,int(D1*self.factor),int(D2*self.factor))

        x_c1 = self.conv1(x_c0 ,D1//2,D2//2)

        x_c2 = self.conv2(x_c1 ,D1//4,D2//4)
    
        x_c3 = self.conv3(x_c2,D1//4,D2//4) 

        x_c4 = self.conv4(x_c3,D1//2,D2//2)
        x_c4 = torch.cat([x_c4, x_c1], dim=1)

        x_c5 = self.conv5(x_c4,int(D1*self.factor),int(D2*self.factor))
        x_c5 = torch.cat([x_c5, x_c0], dim=1)

        x_c6 = self.conv6(x_c5,D1,D2)

        
        if self.padding!=0:
            x_c6 = x_c6[..., 0:-self.padding, 0:-self.padding]

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
