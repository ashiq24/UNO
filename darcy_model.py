#from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *
from Adam import Adam


torch.manual_seed(0)
np.random.seed(0)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, dim1, dim2,modes1 = None, modes2 = None):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim1 = dim1 #output dimensions
        self.dim2 = dim2
        if modes1 is not None:
            self.modes1 = modes1 #Number of Fourier modes to multiply,
            self.modes2 = modes2
        else:
            self.modes1 = dim1//2-1 
            self.modes2 = dim2//2-1
        self.scale = (1 / (2*in_channels))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * (torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))
        self.weights2 = nn.Parameter(self.scale * (torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, dim1 = None,dim2 = None):
        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  self.dim1, self.dim2//2 + 1 , dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(self.dim1, self.dim2))
        return x


class pointwise_op(nn.Module):
    def __init__(self, in_channel, out_channel,dim1, dim2):
        super(pointwise_op,self).__init__()
        self.conv = nn.Conv2d(int(in_channel), int(out_channel), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)

    def forward(self,x, dim1 = None, dim2 = None):
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
        x_out = self.conv(x)
        x_out = torch.nn.functional.interpolate(x_out, size = (dim1, dim2),mode = 'bicubic',align_corners=True)
        return x_out


###############
#  UNO+ achitechtures
###############
class UNO_P_13(nn.Module):
    def __init__(self, in_width, width,pad = 5, factor = 1):
        super(UNO_P_13, self).__init__()

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  # pad the domain if input is non-periodic

        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)

        #self.conv_lin = SpectralConv2d(self.width, self.width,85, 85, 24, 24)

        self.conv0 = SpectralConv2d(self.width, 2*factor*self.width,40, 40, 20, 20)

        self.conv1 = SpectralConv2d(2*factor*self.width, 4*factor*self.width, 20, 20, 10,10)

        self.conv2 = SpectralConv2d(4*factor*self.width, 8*factor*self.width, 10, 10,5,5)
        
        self.conv3 = SpectralConv2d(8*factor*self.width, 16*factor*self.width, 5, 5,3,3)
        
        self.conv4 = SpectralConv2d(16*factor*self.width, 16*factor*self.width, 5, 5,3,3)
        
        self.conv5 = SpectralConv2d(16*factor*self.width, 16*factor*self.width, 5, 5,3,3)
        
        self.conv6 = SpectralConv2d(16*factor*self.width, 16*factor*self.width, 5, 5,3,3)
        
        self.conv7 = SpectralConv2d(16*factor*self.width, 16*factor*self.width, 5, 5,3,3)
        
        self.conv8 = SpectralConv2d(16*factor*self.width, 16*factor*self.width, 5, 5,3,3)
        
        self.conv9 = SpectralConv2d(16*factor*self.width, 8*factor*self.width, 10, 10,3,3)
        
        self.conv10 = SpectralConv2d(16*factor*self.width, 4*factor*self.width, 20, 20,5,5)

        self.conv11 = SpectralConv2d(8*factor*self.width, 2*factor*self.width, 40, 40,10,10)

        self.conv12 = SpectralConv2d(4*factor*self.width, self.width, 85, 85,20,20) # will be reshaped

        #self.w_lin = pointwise_op(self.width,self.width,64, 64)

        self.w0 = pointwise_op(self.width,2*factor*self.width,32, 32) #
        
        self.w1 = pointwise_op(2*factor*self.width, 4*factor*self.width, 16, 16) #
        
        self.w2 = pointwise_op(4*factor*self.width, 8*factor*self.width, 8, 8) #
        
        self.w3 = pointwise_op(8*factor*self.width, 16*factor*self.width, 8, 8)
        
        self.w4 = pointwise_op(16*factor*self.width, 16*factor*self.width, 8, 8)
        
        self.w5 = pointwise_op(16*factor*self.width, 16*factor*self.width, 8, 8)
        
        self.w6 = pointwise_op(16*factor*self.width, 16*factor*self.width, 8, 8)
        
        self.w7 = pointwise_op(16*factor*self.width, 16*factor*self.width, 8, 8)
        
        self.w8 = pointwise_op(16*factor*self.width, 16*factor*self.width, 8, 8)
        
        self.w9 = pointwise_op(16*factor*self.width, 8*factor*self.width, 16, 16)

        self.w10 = pointwise_op(16*factor*self.width, 4*factor*self.width, 16, 16) #
        
        self.w11 = pointwise_op(8*factor*self.width, 2*factor*self.width, 32, 32)
        
        self.w12 = pointwise_op(4*factor*self.width, self.width, 64, 64) # will be reshaped

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

        x1_c0 = self.conv0(x_fc0,D1//2,D2//2)
        x2_c0 = self.w0(x_fc0,D1//2,D2//2)
        x_c0 = x1_c0 + x2_c0
        x_c0 = F.gelu(x_c0)
        #print(x.shape)

        x1_c1 = self.conv1(x_c0,D1//4,D2//4)
        x2_c1 = self.w1(x_c0,D1//4,D2//4)
        x_c1 = x1_c1 + x2_c1
        x_c1 = F.gelu(x_c1)
        #print(x.shape)

        x1_c2 = self.conv2(x_c1,D1//8,D2//8)
        x2_c2 = self.w2(x_c1,D1//8,D2//8)
        x_c2 = x1_c2 + x2_c2
        x_c2 = F.gelu(x_c2)
        #print(x.shape)
        
        x1_c3 = self.conv3(x_c2,D1//16,D2//16)
        x2_c3 = self.w3(x_c2,D1//16,D2//16)
        x_c3 = x1_c3 + x2_c3
        x_c3 = F.gelu(x_c3)
        
        x1_c4 = self.conv4(x_c3,D1//16,D2//16)
        x2_c4 = self.w4(x_c3,D1//16,D2//16)
        x_c4 = x1_c4 + x2_c4
        x_c4 = F.gelu(x_c4)
        
        x1_c5 = self.conv5(x_c4,D1//16,D2//16)
        x2_c5 = self.w5(x_c4,D1//16,D2//16)
        x_c5 = x1_c5 + x2_c5
        x_c5 = F.gelu(x_c5)
        
        x1_c6 = self.conv6(x_c5,D1//16,D2//16)
        x2_c6 = self.w6(x_c5,D1//16,D2//16)
        x_c6 = x1_c6 + x2_c6
        x_c6 = F.gelu(x_c6)
        
        x1_c7 = self.conv7(x_c6,D1//16,D2//16)
        x2_c7 = self.w7(x_c6,D1//16,D2//16)
        x_c7 = x1_c7 + x2_c7
        x_c7 = F.gelu(x_c7)
        
        x1_c8 = self.conv8(x_c7,D1//16,D2//16)
        x2_c8 = self.w8(x_c7,D1//16,D2//16)
        x_c8 = x1_c8 + x2_c8
        x_c8 = F.gelu(x_c8)
        
        x1_c9 = self.conv9(x_c8,D1//8,D2//8)
        x2_c9 = self.w9(x_c8,D1//8,D2//8)
        x_c9 = x1_c9 + x2_c9
        x_c9 = F.gelu(x_c9)
        x_c9 = torch.cat([x_c9, x_c2], dim=1) 
        
        x1_c10 = self.conv10(x_c9 ,D1//4,D2//4)
        x2_c10 = self.w10(x_c9 ,D1//4,D2//4)
        x_c10 = x1_c10 + x2_c10
        x_c10 = F.gelu(x_c10)
        x_c10 = torch.cat([x_c10, x_c1], dim=1)

        x1_c11 = self.conv11(x_c10 ,D1//2,D2//2)
        x2_c11 = self.w11(x_c10 ,D1//2,D2//2)
        x_c11 = x1_c11 + x2_c11
        x_c11 = F.gelu(x_c11)
        x_c11 = torch.cat([x_c11, x_c0], dim=1)

        x1_c12 = self.conv12(x_c11,D1,D2)
        x2_c12 = self.w12(x_c11,D1,D2)
        x_c12 = x1_c12 + x2_c12
        x_c12 = F.gelu(x_c12)
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
##9 layers
    
class UNO_P_9(nn.Module):
    def __init__(self, in_width, width,pad = 5, factor = 1):
        super(UNO_P_9, self).__init__()

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  # pad the domain if input is non-periodic

        self.fc_n1 = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)

        #self.conv_lin = SpectralConv2d(self.width, self.width,85, 85, 24, 24)

        self.conv0 = SpectralConv2d(self.width, 2*factor*self.width,40, 40, 20, 20)

        self.conv1 = SpectralConv2d(2*factor*self.width, 4*factor*self.width, 20, 20, 10,10)

        self.conv2 = SpectralConv2d(4*factor*self.width, 8*factor*self.width, 10, 10,5,5)
        
        self.conv3 = SpectralConv2d(8*factor*self.width, 16*factor*self.width, 5, 5,3,3)
        
        self.conv4 = SpectralConv2d(16*factor*self.width, 16*factor*self.width, 5, 5,3,3)
        
        self.conv5 = SpectralConv2d(16*factor*self.width, 8*factor*self.width, 10, 10,3,3)
        
        self.conv6 = SpectralConv2d(16*factor*self.width, 4*factor*self.width, 20, 20,5,5)

        self.conv7 = SpectralConv2d(8*factor*self.width, 2*factor*self.width, 40, 40,10,10)

        self.conv8 = SpectralConv2d(4*factor*self.width, self.width, 85, 85,20,20) # will be reshaped

        #self.w_lin = pointwise_op(self.width,self.width,64, 64)

        self.w0 = pointwise_op(self.width,2*factor*self.width,32, 32) #
        
        self.w1 = pointwise_op(2*factor*self.width, 4*factor*self.width, 16, 16) #
        
        self.w2 = pointwise_op(4*factor*self.width, 8*factor*self.width, 8, 8) #
        
        self.w3 = pointwise_op(8*factor*self.width, 16*factor*self.width, 8, 8)
        
        self.w4 = pointwise_op(16*factor*self.width, 16*factor*self.width, 8, 8)
        
        self.w5 = pointwise_op(16*factor*self.width, 8*factor*self.width, 16, 16)

        self.w6 = pointwise_op(16*factor*self.width, 4*factor*self.width, 16, 16) #
        
        self.w7 = pointwise_op(8*factor*self.width, 2*factor*self.width, 32, 32)
        
        self.w8 = pointwise_op(4*factor*self.width, self.width, 64, 64) # will be reshaped

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

        x1_c0 = self.conv0(x_fc0,D1//2,D2//2)
        x2_c0 = self.w0(x_fc0,D1//2,D2//2)
        x_c0 = x1_c0 + x2_c0
        x_c0 = F.gelu(x_c0)
        #print(x.shape)

        x1_c1 = self.conv1(x_c0,D1//4,D2//4)
        x2_c1 = self.w1(x_c0,D1//4,D2//4)
        x_c1 = x1_c1 + x2_c1
        x_c1 = F.gelu(x_c1)
        #print(x.shape)

        x1_c2 = self.conv2(x_c1,D1//8,D2//8)
        x2_c2 = self.w2(x_c1,D1//8,D2//8)
        x_c2 = x1_c2 + x2_c2
        x_c2 = F.gelu(x_c2)
        #print(x.shape)
        
        x1_c3 = self.conv3(x_c2,D1//16,D2//16)
        x2_c3 = self.w3(x_c2,D1//16,D2//16)
        x_c3 = x1_c3 + x2_c3
        x_c3 = F.gelu(x_c3)
        
        x1_c4 = self.conv4(x_c3,D1//16,D2//16)
        x2_c4 = self.w4(x_c3,D1//16,D2//16)
        x_c4 = x1_c4 + x2_c4
        x_c4 = F.gelu(x_c4)
 
        x1_c5 = self.conv5(x_c4,D1//8,D2//8)
        x2_c5 = self.w5(x_c4,D1//8,D2//8)
        x_c5 = x1_c5 + x2_c5
        x_c5 = F.gelu(x_c5)
        x_c5 = torch.cat([x_c5, x_c2], dim=1) 
        
        x1_c6 = self.conv6(x_c5 ,D1//4,D2//4)
        x2_c6 = self.w6(x_c5 ,D1//4,D2//4)
        x_c6 = x1_c6 + x2_c6
        x_c6 = F.gelu(x_c6)
        x_c6 = torch.cat([x_c6, x_c1], dim=1)

        x1_c7 = self.conv7(x_c6 ,D1//2,D2//2)
        x2_c7 = self.w7(x_c6 ,D1//2,D2//2)
        x_c7 = x1_c7 + x2_c7
        x_c7 = F.gelu(x_c7)
        x_c7 = torch.cat([x_c7, x_c0], dim=1)

        x1_c8 = self.conv8(x_c7,D1,D2)
        x2_c8 = self.w8(x_c7,D1,D2)
        x_c8 = x1_c8 + x2_c8
        x_c8 = F.gelu(x_c8)
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
    def __init__(self,in_width, width,pad = 6, factor = 3/4):
        super(UNO, self).__init__()
        
        self.in_width = in_width # input channel
        self.width = width 
        self.factor = factor
        self.padding = pad  # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(self.in_width, self.width) # input channel is 3: (a(x, y), x, y)
        
        self.conv0 = SpectralConv2d(self.width, 2*factor*self.width,64, 64, 16, 16)

        self.conv1 = SpectralConv2d(2*factor*self.width, 4*factor*self.width, 32, 32, 12,12)

        self.conv2 = SpectralConv2d(4*factor*self.width, 8*factor*self.width, 16, 16,6,6)
        
        self.conv3 = SpectralConv2d(8*factor*self.width, 8*factor*self.width, 16, 16,6,6)
        
        self.conv4 = SpectralConv2d(8*factor*self.width, 4*factor*self.width, 32, 32,8,8)

        self.conv5 = SpectralConv2d(8*factor*self.width, 2*factor*self.width, 64, 64,12,12)

        self.conv6 = SpectralConv2d(4*factor*self.width, self.width, 85, 85,16,16) # will be reshaped
        
        self.w0 = pointwise_op(self.width,2*factor*self.width,64, 64) #
        
        self.w1 = pointwise_op(2*factor*self.width, 4*factor*self.width, 32, 32) #
        
        self.w2 = pointwise_op(4*factor*self.width, 8*factor*self.width, 16, 16) #
        
        self.w3 = pointwise_op(8*factor*self.width, 8*factor*self.width, 16, 16)
        
        self.w4 = pointwise_op(8*factor*self.width, 4*factor*self.width, 32, 32) #
        
        self.w5 = pointwise_op(8*factor*self.width, 2*factor*self.width, 64, 64)
        
        self.w6 = pointwise_op(4*factor*self.width, self.width, 85, 85) # will be reshaped

        self.fc1 = nn.Linear(1*self.width, 2*self.width)
        self.fc2 = nn.Linear(2*self.width, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)


        x_fc0 = self.fc0(x)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        
        
        x_fc0 = F.pad(x_fc0, [0,self.padding, 0,self.padding])#
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]
        

        x1_c0 = self.conv0(x_fc0,int(D1*self.factor),int(D2*self.factor))
        x2_c0 = self.w0(x_fc0,int(D1*self.factor),int(D2*self.factor))
        x_c0 = x1_c0 + x2_c0
        x_c0 = F.gelu(x_c0)
        #print(x.shape)

        x1_c1 = self.conv1(x_c0 ,D1//2,D2//2)
        x2_c1 = self.w1(x_c0 ,D1//2,D2//2)
        x_c1 = x1_c1 + x2_c1
        x_c1 = F.gelu(x_c1)
        #print(x.shape)

        x1_c2 = self.conv2(x_c1 ,D1//4,D2//4)
        x2_c2 = self.w2(x_c1 ,D1//4,D2//4)
        x_c2 = x1_c2 + x2_c2
        x_c2 = F.gelu(x_c2 )
        #print(x.shape)
             
        x1_c3 = self.conv3(x_c2,D1//4,D2//4)
        x2_c3 = self.w3(x_c2,D1//4,D2//4)
        x_c3 = x1_c3 + x2_c3
        x_c3 = F.gelu(x_c3) 

        x1_c4 = self.conv4(x_c3,D1//2,D2//2)
        x2_c4 = self.w4(x_c3,D1//2,D2//2)
        x_c4 = x1_c4 + x2_c4
        x_c4 = F.gelu(x_c4)
        x_c4 = torch.cat([x_c4, x_c1], dim=1)

        x1_c5 = self.conv5(x_c4,int(D1*self.factor),int(D2*self.factor))
        x2_c5 = self.w5(x_c4,int(D1*self.factor),int(D2*self.factor))
        x_c5 = x1_c5 + x2_c5
        x_c5 = F.gelu(x_c5)
        x_c5 = torch.cat([x_c5, x_c0], dim=1)

        x1_c6 = self.conv6(x_c5,D1,D2)
        x2_c6 = self.w6(x_c5,D1,D2)
        x_c6 = x1_c6 + x2_c6
        x_c6 = F.gelu(x_c6)
        
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