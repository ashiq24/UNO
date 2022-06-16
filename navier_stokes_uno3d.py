# Codes for section: Results on Navier Stocks Equation (3D)
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from integral_operators import *
import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)



class Uno3D_T40(nn.Module):
    """
    The overall network. It contains 4 layers of the Fourier layer.
    1. Lift the input to the desire channel dimension by  self.fc, self.fc0 .
    2. 4 layers of the integral operators u' = (W + K)(u).
        W defined by self.w; K defined by self.conv .
    3. Project from the channel space to the output space by self.fc1 and self.fc2 .
    
    input: the solution of the first 10 timesteps (u(1), ..., u(10)).
    input shape: (batchsize, x=S, y=S, t=T, c=1)
    output: the solution of the next 40 timesteps
    output shape: (batchsize, x=S, y=S, t=4*T, c=1)
    S,S,T = grid size along x,y and t (of input function)
    S,S,4*T = grid size along x,y and t (of output function)
    (Note that model is dicretization invarient in both spatial (x,y) and time (t) domain)
    
    in_width = 4; [a(x,y,x),x,y,z]
    with = uplifting dimesion
    pad = padding amount
    pad_both = boolean, if true pad both size of the domain
    factor = scaling factor of the co-domain dimesions 
    """
    def __init__(self, in_width, width,pad = 1, factor = 1, pad_both = False):
        super(Uno3D_T40, self).__init__()

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  
        self.pad_both = pad_both

        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width)  
        
        self.conv0 = OperatorBlock_3D(self.width, 2*factor*self.width,48, 48, 10, 18,18, 4)
        
        self.conv1 = OperatorBlock_3D(2*factor*self.width, 4*factor*self.width, 32, 32,10,  14,14,4)
        
        self.conv2 = OperatorBlock_3D(4*factor*self.width, 8*factor*self.width, 16, 16, 16,  6,6,4)
        
        self.conv3 = OperatorBlock_3D(8*factor*self.width, 16*factor*self.width, 8, 8, 16,  3,3,7)
        
        self.conv4 = OperatorBlock_3D(16*factor*self.width, 16*factor*self.width, 8, 8, 16,  3,3,7)
        
        self.conv5 = OperatorBlock_3D(16*factor*self.width, 8*factor*self.width, 16, 16, 16,  3,3,7) 
        
        self.conv6 = OperatorBlock_3D(8*factor*self.width, 4*factor*self.width, 32, 32, 24,  6,6,7)
        
        self.conv7 = OperatorBlock_3D(8*factor*self.width, 2*factor*self.width, 48, 48, 32,  14,14,10)
        
        self.conv8 = OperatorBlock_3D(4*factor*self.width, 2*self.width, 64, 64, 40,  18,18, 14) # will be reshaped

        self.fc1 = nn.Linear(3*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 1)
        


    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x_fc_1 = self.fc(x)
        x_fc_1 = F.gelu(x_fc_1)
        x_fc0 = self.fc0(x_fc_1)

        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)
        
        if self.pad_both:
            x_fc0 = F.pad(x_fc0, [self.padding,self.padding,0,0,0,0],mode ='constant')
        else:
            x_fc0 = F.pad(x_fc0, [0,self.padding,0,0,0,0],mode ='constant')
        
        D1,D2,D3 = x_fc0.shape[-3],x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.conv0(x_fc0, int(3*D1/4),int(3*D2/4),D3)
        x_c1 = self.conv1(x_c0, D1//2, D2//2, D3)
        x_c2 = self.conv2(x_c1, D1//4, D2//4, int(D3*1.6))
        
        x_c3 = self.conv3(x_c2, D1//8, D2//8, int(D3*1.6))
        x_c4 = self.conv4(x_c3, D1//8, D2//8, int(D3*1.6))
        x_c5 = self.conv5(x_c4, D1//4, D2//4, int(D3*1.6))
        

        x_c6 = self.conv6(x_c5,D1//2, D2//2, int(D3*2.4))
        x_c6 = torch.cat([x_c6, torch.nn.functional.interpolate(x_c1, size = (x_c6.shape[2], x_c6.shape[3],x_c6.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
        

        x_c7 = self.conv7(x_c6, int(3*D1/4),int(3*D2/4),int(3.2*D3))
        x_c7 = torch.cat([x_c7, torch.nn.functional.interpolate(x_c0, size = (x_c7.shape[2], x_c7.shape[3],x_c7.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
        

        
        x_c8 = self.conv8(x_c7,D1,D2,4*D3)

        x_c8 = torch.cat([x_c8,torch.nn.functional.interpolate(x_fc0, size = (x_c8.shape[2], x_c8.shape[3],x_c8.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)

        
        if self.padding!=0:
            if self.pad_both:
                x_c8 = x_c8[...,4*self.padding:-4*self.padding]
            else:
                x_c8 = x_c8[...,:-4*self.padding]

        x_c8 = x_c8.permute(0, 2, 3, 4, 1)

        x_fc1 = self.fc1(x_c8)

        x_fc1 = F.gelu(x_fc1)
        x_out = self.fc2(x_fc1)
        
        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

#########
# Time = 20
# ########   
class Uno3D_T20(nn.Module):
    """
    The overall network. It contains 4 layers of the Fourier layer.
    1. Lift the input to the desire channel dimension by  self.fc, self.fc0 .
    2. 4 layers of the integral operators u' = (W + K)(u).
        W defined by self.w; K defined by self.conv .
    3. Project from the channel space to the output space by self.fc1 and self.fc2 .
    
    input: the solution of the first 10 timesteps (u(1), ..., u(10)).
    input shape: (batchsize, x=S, y=S, t=T, c=1)
    output: the solution of the next 20 timesteps
    output shape: (batchsize, x=S, y=S, t=2*T, c=1)
    
    S,S,T = grid size along x,y and t (input function)
    S,S,2*T = grid size along x,y and t (output function)
    
    in_width = 4; [a(x,y,x),x,y,z]
    with = projection dimesion
    pad = padding amount
    pad_both = boolean, if true pad both size of the domain
    factor = scaling factor of the co-domain dimesions 
    """
    def __init__(self, in_width, width,pad = 2, factor = 1, pad_both = False):
        super(Uno3D_T20, self).__init__()

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  
        self.pad_both = pad_both

        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width)  
        
        self.conv0 = OperatorBlock_3D(self.width, 2*factor*self.width,48, 48, 10, 18,18, 5)
        
        self.conv1 = OperatorBlock_3D(2*factor*self.width, 4*factor*self.width, 32, 32,10,  14,14,5)
        
        self.conv2 = OperatorBlock_3D(4*factor*self.width, 8*factor*self.width, 16, 16, 12,  6,6,5)
        
        self.conv3 = OperatorBlock_3D(8*factor*self.width, 16*factor*self.width, 8, 8, 12,  3,3,6)
        
        self.conv4 = OperatorBlock_3D(16*factor*self.width, 16*factor*self.width, 8, 8, 16,  3,3,6)
        
        self.conv5 = OperatorBlock_3D(16*factor*self.width, 8*factor*self.width, 16, 16, 16,  3,3,8) 
        
        self.conv6 = OperatorBlock_3D(8*factor*self.width, 4*factor*self.width, 32, 32, 18,  6,6,8)
        
        self.conv7 = OperatorBlock_3D(8*factor*self.width, 2*factor*self.width, 48, 48, 20,  14,14,8)
        
        self.conv8 = OperatorBlock_3D(4*factor*self.width, 2*self.width, 64, 64, 20,  18,18, 8) # will be reshaped

        self.fc1 = nn.Linear(3*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 1)
        


    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x_fc_1 = self.fc(x)
        x_fc_1 = F.gelu(x_fc_1)
        x_fc0 = self.fc0(x_fc_1)

        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)
        
        if self.pad_both:
            x_fc0 = F.pad(x_fc0, [self.padding,self.padding,0,0,0,0],mode ='constant')
        else:
            x_fc0 = F.pad(x_fc0, [0,self.padding,0,0,0,0],mode ='constant')
        
        D1,D2,D3 = x_fc0.shape[-3],x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.conv0(x_fc0, int(3*D1/4),int(3*D2/4),D3)
        x_c1 = self.conv1(x_c0, D1//2, D2//2, D3)
        x_c2 = self.conv2(x_c1, D1//4, D2//4, int(D3*1.2))
        
        x_c3 = self.conv3(x_c2, D1//8, D2//8, int(D3*1.2))
        x_c4 = self.conv4(x_c3, D1//8, D2//8, int(D3*1.6))
        x_c5 = self.conv5(x_c4, D1//4, D2//4, int(D3*1.6))
        

        x_c6 = self.conv6(x_c5,D1//2, D2//2, int(D3*1.8))
        x_c6 = torch.cat([x_c6, torch.nn.functional.interpolate(x_c1, size = (x_c6.shape[2], x_c6.shape[3],x_c6.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
        

        x_c7 = self.conv7(x_c6, int(3*D1/4),int(3*D2/4),int(2.0*D3))
        x_c7 = torch.cat([x_c7, torch.nn.functional.interpolate(x_c0, size = (x_c7.shape[2], x_c7.shape[3],x_c7.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
        

        
        x_c8 = self.conv8(x_c7,D1,D2,2*D3)

        x_c8 = torch.cat([x_c8,torch.nn.functional.interpolate(x_fc0, size = (x_c8.shape[2], x_c8.shape[3],x_c8.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)

        
        if self.padding!=0:
            if self.pad_both:
                x_c8 = x_c8[...,2*self.padding:-2*self.padding]
            else:
                x_c8 = x_c8[...,:-2*self.padding]

        x_c8 = x_c8.permute(0, 2, 3, 4, 1)

        x_fc1 = self.fc1(x_c8)

        x_fc1 = F.gelu(x_fc1)
        x_out = self.fc2(x_fc1)
        
        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

    
class Uno3D_T10(nn.Module):
    """
    The overall network. It contains 4 layers of the Fourier layer.
    1. Lift the input to the desire channel dimension by  self.fc, self.fc0 .
    2. 4 layers of the integral operators u' = (W + K)(u).
        W defined by self.w; K defined by self.conv .
    3. Project from the channel space to the output space by self.fc1 and self.fc2 .
    
    input: the solution of the first 10 timesteps (u(1), ..., u(10)).
    input shape: (batchsize, x=S, y=S, t=T, c=1)
    output: the solution of the next 10 timesteps
    output shape: (batchsize, x=S, y=S, t=T, c=1)
    
    S,S,T = grid size along x,y and t (input function)
    S,S,T = grid size along x,y and t (output function)
    
    in_width = 4; [a(x,y,x),x,y,z]
    with = projection dimesion
    pad = padding amount
    pad_both = boolean, if true pad both size of the domain
    factor = scaling factor of the co-domain dimesions 
    """
    def __init__(self, in_width, width,pad = 2, factor = 1, pad_both = False):
        super(Uno3D_T10, self).__init__()

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  
        self.pad_both = pad_both

        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width)  
        
        self.conv0 = OperatorBlock_3D(self.width, 2*factor*self.width,48, 48, 10, 18,18, 4)
        
        self.conv1 = OperatorBlock_3D(2*factor*self.width, 4*factor*self.width, 32, 32,10,  14,14,4)
        
        self.conv2 = OperatorBlock_3D(4*factor*self.width, 8*factor*self.width, 16, 16, 8,  6,6,3)
        
        self.conv3 = OperatorBlock_3D(8*factor*self.width, 16*factor*self.width, 8, 8, 8,  3,3,3)
        
        self.conv4 = OperatorBlock_3D(16*factor*self.width, 16*factor*self.width, 8, 8, 8,  3,3,3)
        
        self.conv5 = OperatorBlock_3D(16*factor*self.width, 8*factor*self.width, 16, 16, 8,  3,3,3) 
        
        self.conv6 = OperatorBlock_3D(8*factor*self.width, 4*factor*self.width, 32, 32, 8,  6,6,3)
        
        self.conv7 = OperatorBlock_3D(8*factor*self.width, 2*factor*self.width, 48, 48, 10,  14,14,3)
        
        self.conv8 = OperatorBlock_3D(4*factor*self.width, 2*self.width, 64, 64, 10,  18,18, 4) # will be reshaped

        self.fc1 = nn.Linear(3*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 1)
        


    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x_fc_1 = self.fc(x)
        x_fc_1 = F.gelu(x_fc_1)
        x_fc0 = self.fc0(x_fc_1)

        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)
        
        if self.pad_both:
            x_fc0 = F.pad(x_fc0, [self.padding,self.padding,0,0,0,0],mode ='constant')
        else:
            x_fc0 = F.pad(x_fc0, [0,self.padding,0,0,0,0],mode ='constant')
        
        D1,D2,D3 = x_fc0.shape[-3],x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.conv0(x_fc0, int(3*D1/4),int(3*D2/4),D3)
        x_c1 = self.conv1(x_c0, D1//2, D2//2, D3)
        x_c2 = self.conv2(x_c1, D1//4, D2//4, int(0.8*D3))
        
        x_c3 = self.conv3(x_c2, D1//8, D2//8, int(0.8*D3))
        x_c4 = self.conv4(x_c3, D1//8, D2//8, int(0.8*D3))
        x_c5 = self.conv5(x_c4, D1//4, D2//4, int(0.8*D3))
        

        x_c6 = self.conv6(x_c5,D1//2, D2//2, int(0.8*D3))
        x_c6 = torch.cat([x_c6, torch.nn.functional.interpolate(x_c1, size = (x_c6.shape[2], x_c6.shape[3],x_c6.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
        

        x_c7 = self.conv7(x_c6, int(3*D1/4),int(3*D2/4),D3)
        x_c7 = torch.cat([x_c7, torch.nn.functional.interpolate(x_c0, size = (x_c7.shape[2], x_c7.shape[3],x_c7.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
        

        
        x_c8 = self.conv8(x_c7,D1,D2,D3)

        x_c8 = torch.cat([x_c8,torch.nn.functional.interpolate(x_fc0, size = (x_c8.shape[2], x_c8.shape[3],x_c8.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)

        
        if self.padding!=0:
            if self.pad_both:
                x_c8 = x_c8[...,self.padding:-self.padding]
            else:
                x_c8 = x_c8[...,:-self.padding]

        x_c8 = x_c8.permute(0, 2, 3, 4, 1)

        x_fc1 = self.fc1(x_c8)

        x_fc1 = F.gelu(x_fc1)
        x_out = self.fc2(x_fc1)
        
        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

    
class Uno3D_T9(nn.Module):
    """
    The overall network. It contains 4 layers of the Fourier layer.
    1. Lift the input to the desire channel dimension by  self.fc, self.fc0 .
    2. 4 layers of the integral operators u' = (W + K)(u).
        W defined by self.w; K defined by self.conv .
    3. Project from the channel space to the output space by self.fc1 and self.fc2 .
    
    input: the solution of the first 6 timesteps (u(1), ..., u(10)).
    input shape: (batchsize, x=S, y=S, t=T, c=1)
    output: the solution of the next 9 timesteps
    output shape: (batchsize, x=S, y=S, t=int(1.5*T), c=1)
    
    S,S,T = grid size along x,y and t (input function)
    S,S,int(1.5*T) = grid size along x,y and t (output function)
    
    in_width = 4; [a(x,y,x),x,y,z]
    with = projection dimesion
    pad = padding amount
    pad_both = boolean, if true pad both size of the domain
    factor = scaling factor of the co-domain dimesions 
    """
    def __init__(self, in_width, width,pad = 2, factor = 1, pad_both = False):
        super(Uno3D_T9, self).__init__()

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  
        self.pad_both = pad_both

        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width)  
        
        self.conv0 = OperatorBlock_3D(self.width, 2*factor*self.width,48, 48, 6, 18,18, 3)
        
        self.conv1 = OperatorBlock_3D(2*factor*self.width, 4*factor*self.width, 32, 32,6,  18,18,3)
        
        self.conv2 = OperatorBlock_3D(4*factor*self.width, 8*factor*self.width, 16, 16, 6,  6,6,3)
        
        self.conv3 = OperatorBlock_3D(8*factor*self.width, 16*factor*self.width, 8, 8, 8,  3,3,3)
        
        self.conv4 = OperatorBlock_3D(16*factor*self.width, 16*factor*self.width, 8, 8, 8,  3,3,3)
        
        self.conv5 = OperatorBlock_3D(16*factor*self.width, 8*factor*self.width, 16, 16, 8,  3,3,3) 
        
        self.conv6 = OperatorBlock_3D(8*factor*self.width, 4*factor*self.width, 32, 32, 8,  6,6,3)
        
        self.conv7 = OperatorBlock_3D(8*factor*self.width, 2*factor*self.width, 48, 48, 9,  14,14,3)
        
        self.conv8 = OperatorBlock_3D(4*factor*self.width, 2*self.width, 64, 64, 9,  18,18, 4) # will be reshaped

        self.fc1 = nn.Linear(3*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 1)
        


    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x_fc_1 = self.fc(x)
        x_fc_1 = F.gelu(x_fc_1)
        x_fc0 = self.fc0(x_fc_1)

        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)
        
        if self.pad_both:
            x_fc0 = F.pad(x_fc0, [self.padding,self.padding,0,0,0,0],mode ='constant')
        else:
            x_fc0 = F.pad(x_fc0, [0,self.padding,0,0,0,0],mode ='constant')
        
        D1,D2,D3 = x_fc0.shape[-3],x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.conv0(x_fc0, int(3*D1/4),int(3*D2/4),D3)
        x_c1 = self.conv1(x_c0, D1//2, D2//2, D3)
        x_c2 = self.conv2(x_c1, D1//4, D2//4, D3)
        
        x_c3 = self.conv3(x_c2, D1//8, D2//8, int(8*D3/6))
        x_c4 = self.conv4(x_c3, D1//8, D2//8, int(8*D3/6))
        x_c5 = self.conv5(x_c4, D1//4, D2//4, int(8*D3/6))
        

        x_c6 = self.conv6(x_c5,D1//2, D2//2, int(8*D3/6))
        x_c6 = torch.cat([x_c6, torch.nn.functional.interpolate(x_c1, size = (x_c6.shape[2], x_c6.shape[3],x_c6.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
        

        x_c7 = self.conv7(x_c6, int(3*D1/4),int(3*D2/4), int(9*D3/6))
        x_c7 = torch.cat([x_c7, torch.nn.functional.interpolate(x_c0, size = (x_c7.shape[2], x_c7.shape[3],x_c7.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
        

        
        x_c8 = self.conv8(x_c7,D1,D2, int(9*D3/6))

        x_c8 = torch.cat([x_c8,torch.nn.functional.interpolate(x_fc0, size = (x_c8.shape[2], x_c8.shape[3],x_c8.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)

        
        if self.padding!=0:
            if self.pad_both:
                x_c8 = x_c8[...,int(9*self.padding/6):-int(9*self.padding/6)]
            else:
                x_c8 = x_c8[...,:-int(9*self.padding/6)]

        x_c8 = x_c8.permute(0, 2, 3, 4, 1)

        x_fc1 = self.fc1(x_c8)

        x_fc1 = F.gelu(x_fc1)
        x_out = self.fc2(x_fc1)
        
        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


######
# Codes for Higher resolution simulation of Navier Stockes equations. Functionality of each 
# of the following classes are as the above classes. 
######

class Uno3D_T40_256(nn.Module):
    def __init__(self, in_width, width,pad = 1, factor = 1):
        super(Uno3D_T40_256, self).__init__()

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  

        self.fc_n1 = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width)  
        
        self.conv0 = OperatorBlock_3D(self.width, 2*factor*self.width,64, 64, 10, 32,32, 5)
        
        self.conv1 = OperatorBlock_3D(2*factor*self.width, 4*factor*self.width, 16, 16,10,  8,8,5)
        
        self.conv2 = OperatorBlock_3D(4*factor*self.width, 8*factor*self.width, 8, 8, 16,  4,4,5)
        
        self.conv3 = OperatorBlock_3D(8*factor*self.width, 16*factor*self.width, 8, 8, 16,  4,4,8)
        
        self.conv4 = OperatorBlock_3D(16*factor*self.width, 16*factor*self.width, 8, 8, 16,  4,4,8)
        
        self.conv5 = OperatorBlock_3D(16*factor*self.width, 8*factor*self.width, 8, 8, 16,  4,4,8) 
        
        self.conv6 = OperatorBlock_3D(8*factor*self.width, 4*factor*self.width, 16, 16, 24,  4,4,8)
        
        self.conv7 = OperatorBlock_3D(8*factor*self.width, 2*factor*self.width, 64, 64, 32,  8,8,12)
        
        self.conv8 = OperatorBlock_3D(4*factor*self.width, 2*self.width, 256, 256, 40,  32,32, 16) # will be reshaped

        self.fc1 = nn.Linear(3*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 1)
        


    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x_fc_1 = self.fc(x)
        x_fc_1 = F.gelu(x_fc_1)
        x_fc0 = self.fc0(x_fc_1)

        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)
        
        if self.pad_both:
            x_fc0 = F.pad(x_fc0, [self.padding,self.padding,0,0,0,0],mode ='constant')
        else:
            x_fc0 = F.pad(x_fc0, [0,self.padding,0,0,0,0],mode ='constant')
        
        D1,D2,D3 = x_fc0.shape[-3],x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.conv0(x_fc0, D1//4, D2//4,D3)
        x_c1 = self.conv1(x_c0, D1//16, D2//16, D3)
        x_c2 = self.conv2(x_c1, D1//32, D2//32, int(D3*1.6))
        
        x_c3 = self.conv3(x_c2,  D1//32, D2//32, int(D3*1.6))
        x_c4 = self.conv4(x_c3,  D1//32, D2//32, int(D3*1.6))
        x_c5 = self.conv5(x_c4,  D1//32, D2//32, int(D3*1.6))
        

        x_c6 = self.conv6(x_c5,  D1//16, D2//16, int(D3*2.4))
        x_c6 = torch.cat([x_c6, torch.nn.functional.interpolate(x_c1, size = (x_c6.shape[2], x_c6.shape[3],x_c6.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
        

        x_c7 = self.conv7(x_c6,  D1//4, D2//4,int(3.2*D3))
        x_c7 = torch.cat([x_c7, torch.nn.functional.interpolate(x_c0, size = (x_c7.shape[2], x_c7.shape[3],x_c7.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
        

        
        x_c8 = self.conv8(x_c7,D1,D2,4*D3)

        x_c8 = torch.cat([x_c8,torch.nn.functional.interpolate(x_fc0, size = (x_c8.shape[2], x_c8.shape[3],x_c8.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)

        
        if self.padding!=0:
            if self.pad_both:
                x_c8 = x_c8[...,4*self.padding:-4*self.padding]
            else:
                x_c8 = x_c8[...,:-4*self.padding]

        x_c8 = x_c8.permute(0, 2, 3, 4, 1)

        x_fc1 = self.fc1(x_c8)

        x_fc1 = F.gelu(x_fc1)
        x_out = self.fc2(x_fc1)
        
        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


class Uno3D_T20_256(nn.Module):
    def __init__(self, in_width, width,pad = 2, factor = 1, pad_both = False):
        super(Uno3D_T20_256, self).__init__()

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  
        self.pad_both = pad_both

        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width)  
        
        self.conv0 = OperatorBlock_3D(self.width, 2*factor*self.width,64, 64, 10, 32,32, 5)
        
        self.conv1 = OperatorBlock_3D(2*factor*self.width, 4*factor*self.width, 16, 16,10,  8,8,5)
        
        self.conv2 = OperatorBlock_3D(4*factor*self.width, 8*factor*self.width, 8, 8, 12,  4,4,5)
        
        self.conv3 = OperatorBlock_3D(8*factor*self.width, 16*factor*self.width, 8, 8, 12,  4,4,6)
        
        self.conv4 = OperatorBlock_3D(16*factor*self.width, 16*factor*self.width, 8, 8, 16,  4,4,6)
        
        self.conv5 = OperatorBlock_3D(16*factor*self.width, 8*factor*self.width, 8, 8, 16,  4,4,8) 
        
        self.conv6 = OperatorBlock_3D(8*factor*self.width, 4*factor*self.width, 16, 16, 18,  4,4,8)
        
        self.conv7 = OperatorBlock_3D(8*factor*self.width, 2*factor*self.width, 64, 64, 20,  8,8,8)
        
        self.conv8 = OperatorBlock_3D(4*factor*self.width, 2*self.width, 256, 256, 20,  32,32, 8) # will be reshaped

        self.fc1 = nn.Linear(3*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 1)
        


    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x_fc_1 = self.fc(x)
        x_fc_1 = F.gelu(x_fc_1)
        x_fc0 = self.fc0(x_fc_1)

        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)
        
        if self.pad_both:
            x_fc0 = F.pad(x_fc0, [self.padding,self.padding,0,0,0,0],mode ='constant')
        else:
            x_fc0 = F.pad(x_fc0, [0,self.padding,0,0,0,0],mode ='constant')
        
        D1,D2,D3 = x_fc0.shape[-3],x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.conv0(x_fc0, D1//4, D2//4,D3)
        x_c1 = self.conv1(x_c0, D1//16, D2//16, D3)
        x_c2 = self.conv2(x_c1, D1//32, D2//32, int(D3*1.2))
        
        x_c3 = self.conv3(x_c2, D1//32, D2//32, int(D3*1.2))
        x_c4 = self.conv4(x_c3, D1//32, D2//32, int(D3*1.6))
        x_c5 = self.conv5(x_c4, D1//32, D2//32, int(D3*1.6))
        

        x_c6 = self.conv6(x_c5, D1//16, D2//16, int(D3*1.8))
        x_c6 = torch.cat([x_c6, torch.nn.functional.interpolate(x_c1, size = (x_c6.shape[2], x_c6.shape[3],x_c6.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
        

        x_c7 = self.conv7(x_c6, D1//4, D2//4,int(2.0*D3))
        x_c7 = torch.cat([x_c7, torch.nn.functional.interpolate(x_c0, size = (x_c7.shape[2], x_c7.shape[3],x_c7.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
        

        
        x_c8 = self.conv8(x_c7,D1,D2,2*D3)

        x_c8 = torch.cat([x_c8,torch.nn.functional.interpolate(x_fc0, size = (x_c8.shape[2], x_c8.shape[3],x_c8.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)

        
        if self.padding!=0:
            if self.pad_both:
                x_c8 = x_c8[...,2*self.padding:-2*self.padding]
            else:
                x_c8 = x_c8[...,:-2*self.padding]

        x_c8 = x_c8.permute(0, 2, 3, 4, 1)

        x_fc1 = self.fc1(x_c8)

        x_fc1 = F.gelu(x_fc1)
        x_out = self.fc2(x_fc1)
        
        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


class Uno3D_T10_256(nn.Module):
    def __init__(self, in_width, width,pad = 2, factor = 1, pad_both = False):
        super(Uno3D_T10_256, self).__init__()

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  
        self.pad_both = pad_both

        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width)  
        
        self.conv0 = OperatorBlock_3D(self.width, 2*factor*self.width,64, 64, 10, 32,32, 5)
        
        self.conv1 = OperatorBlock_3D(2*factor*self.width, 4*factor*self.width, 16, 16,10,  8,8,4)
        
        self.conv2 = OperatorBlock_3D(4*factor*self.width, 8*factor*self.width, 8, 8, 8,  4,4,4)
        
        self.conv3 = OperatorBlock_3D(8*factor*self.width, 16*factor*self.width, 8, 8, 8,  4,4,4)
        
        self.conv4 = OperatorBlock_3D(16*factor*self.width, 16*factor*self.width, 8, 8, 8,  4,4,4)
        
        self.conv5 = OperatorBlock_3D(16*factor*self.width, 8*factor*self.width, 8, 8, 8,  4,4,4) 
        
        self.conv6 = OperatorBlock_3D(8*factor*self.width, 4*factor*self.width, 16, 16, 8,  4,4,4)
        
        self.conv7 = OperatorBlock_3D(8*factor*self.width, 2*factor*self.width, 64, 64, 10,  8,8,4)
        
        self.conv8 = OperatorBlock_3D(4*factor*self.width, 2*self.width, 256, 256, 10,  32,32, 5) # will be reshaped

        self.fc1 = nn.Linear(3*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 1)
        


    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x_fc_1 = self.fc(x)
        x_fc_1 = F.gelu(x_fc_1)
        x_fc0 = self.fc0(x_fc_1)

        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)
        
        if self.pad_both:
            x_fc0 = F.pad(x_fc0, [self.padding,self.padding,0,0,0,0],mode ='constant')
        else:
            x_fc0 = F.pad(x_fc0, [0,self.padding,0,0,0,0],mode ='constant')
        
        D1,D2,D3 = x_fc0.shape[-3],x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.conv0(x_fc0, D1//4, D2//4,D3)
        x_c1 = self.conv1(x_c0, D1//16, D2//16, D3)
        x_c2 = self.conv2(x_c1, D1//32, D2//32, int(0.8*D3))
        
        x_c3 = self.conv3(x_c2, D1//32, D2//32, int(0.8*D3))
        x_c4 = self.conv4(x_c3, D1//32, D2//32, int(0.8*D3))
        x_c5 = self.conv5(x_c4, D1//32, D2//32, int(0.8*D3))
        

        x_c6 = self.conv6(x_c5, D1//16, D2//16, int(0.8*D3))
        x_c6 = torch.cat([x_c6, torch.nn.functional.interpolate(x_c1, size = (x_c6.shape[2], x_c6.shape[3],x_c6.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
        

        x_c7 = self.conv7(x_c6, D1//4, D2//4,D3)
        x_c7 = torch.cat([x_c7, torch.nn.functional.interpolate(x_c0, size = (x_c7.shape[2], x_c7.shape[3],x_c7.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
        

        
        x_c8 = self.conv8(x_c7,D1,D2,D3)

        x_c8 = torch.cat([x_c8,torch.nn.functional.interpolate(x_fc0, size = (x_c8.shape[2], x_c8.shape[3],x_c8.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)

        
        if self.padding!=0:
            if self.pad_both:
                x_c8 = x_c8[...,self.padding:-self.padding]
            else:
                x_c8 = x_c8[...,:-self.padding]

        x_c8 = x_c8.permute(0, 2, 3, 4, 1)

        x_fc1 = self.fc1(x_c8)

        x_fc1 = F.gelu(x_fc1)
        x_out = self.fc2(x_fc1)
        
        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


class Uno3D_T9_256(nn.Module):
    def __init__(self, in_width, width,pad = 2, factor = 1, pad_both = False):
        super(Uno3D_T10_256, self).__init__()

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  
        self.pad_both = pad_both

        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width)  
        
        self.conv0 = OperatorBlock_3D(self.width, 2*factor*self.width,64, 64, 6, 32,32, 3)
        
        self.conv1 = OperatorBlock_3D(2*factor*self.width, 4*factor*self.width, 16, 16,6,  8,8,3)
        
        self.conv2 = OperatorBlock_3D(4*factor*self.width, 8*factor*self.width, 8, 8, 6,  4,4,3)
        
        self.conv3 = OperatorBlock_3D(8*factor*self.width, 16*factor*self.width, 8, 8, 8,  4,4,3)
        
        self.conv4 = OperatorBlock_3D(16*factor*self.width, 16*factor*self.width, 8, 8, 8,  4,4,4)
        
        self.conv5 = OperatorBlock_3D(16*factor*self.width, 8*factor*self.width, 8, 8, 8,  4,4,4) 
        
        self.conv6 = OperatorBlock_3D(8*factor*self.width, 4*factor*self.width, 16, 16, 8,  4,4,4)
        
        self.conv7 = OperatorBlock_3D(8*factor*self.width, 2*factor*self.width, 64, 64, 9,  4,4,4)
        
        self.conv8 = OperatorBlock_3D(4*factor*self.width, 2*self.width, 256, 256, 9,  32,32, 4) # will be reshaped

        self.fc1 = nn.Linear(3*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 1)
        


    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x_fc_1 = self.fc(x)
        x_fc_1 = F.gelu(x_fc_1)
        x_fc0 = self.fc0(x_fc_1)

        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)
        
        if self.pad_both:
            x_fc0 = F.pad(x_fc0, [self.padding,self.padding,0,0,0,0],mode ='constant')
        else:
            x_fc0 = F.pad(x_fc0, [0,self.padding,0,0,0,0],mode ='constant')
        
        D1,D2,D3 = x_fc0.shape[-3],x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.conv0(x_fc0, D1//4, D2//4,D3)
        x_c1 = self.conv1(x_c0, D1//16, D2//16, D3)
        x_c2 = self.conv2(x_c1, D1//32, D2//32, D3)
        
        x_c3 = self.conv3(x_c2, D1//32, D2//32, int(8*D3/6))
        x_c4 = self.conv4(x_c3, D1//32, D2//32, int(8*D3/6))
        x_c5 = self.conv5(x_c4, D1//32, D2//32, int(8*D3/6))
        

        x_c6 = self.conv6(x_c5,D1//16, D2//16, int(8*D3/6))
        x_c6 = torch.cat([x_c6, torch.nn.functional.interpolate(x_c1, size = (x_c6.shape[2], x_c6.shape[3],x_c6.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
        

        x_c7 = self.conv7(x_c6, D1//4, D2//4, int(9*D3/6))
        x_c7 = torch.cat([x_c7, torch.nn.functional.interpolate(x_c0, size = (x_c7.shape[2], x_c7.shape[3],x_c7.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
        

        
        x_c8 = self.conv8(x_c7,  D1,D2, int(9*D3/6))

        x_c8 = torch.cat([x_c8,torch.nn.functional.interpolate(x_fc0, size = (x_c8.shape[2], x_c8.shape[3],x_c8.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)

        
        if self.padding!=0:
            if self.pad_both:
                x_c8 = x_c8[...,int(9*self.padding/6):-int(9*self.padding/6)]
            else:
                x_c8 = x_c8[...,:-int(9*self.padding/6)]

        x_c8 = x_c8.permute(0, 2, 3, 4, 1)

        x_fc1 = self.fc1(x_c8)

        x_fc1 = F.gelu(x_fc1)
        x_out = self.fc2(x_fc1)
        
        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
