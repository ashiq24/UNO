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
        in_channels = Number of input functions (equivalent to number of input channels)
        out_channels = Number of out functions (equivalent to number of output channels)
        dim1, dim2 = Desired output grid size (dim1xdim2)
        modes1, modes2 = number of fourier modes to consider for 2D fourier Kernel, at most floor(min(dim1,dim2)/2)
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
            self.modes1 = dim1//2 - 1  #if not given take the highest number of modes can be taken
            self.modes2 = dim2//2 
        self.scale = (1 / (in_channels + out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, dim1 = None,dim2 = None):
        '''
        dim1,dim2 = Desired output grid size (dim1xdim2).

        Please note that, test data can be of any grid size (any resolution). The forward function take the derised gride size and 
        do interpolation in the fourier domain to match the desired gride size.
        Please note: this implementation can not handle any coarser grid (any lower resolution) i.e. it can not handle arbitary low
        resolution data as input. But can work with any arbitary finer grid (higher resolution).
        '''
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
        """
        n_channels = Number of input functions (equivalent to number of input channels)
        out_channels = Number of out functions (equivalent to number of output channels)
        dim1, dim2 = Desired output grid size (dim1xdim2)
        """
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

class FNO2d_UNO(nn.Module):
    def __init__(self, in_width, width):
        super(FNO2d_UNO, self).__init__()

        """
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. Repeated layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.in_width = in_width # input channel
        self.width = width 

        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(self.in_width, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, 2*self.width,32, 32, 16, 17) 
        # Number of fourier modes (16,17) is a hyper parameter
        # also the output grid size (32x32) is a design choice

        self.conv1 = SpectralConv2d(2*self.width, 4*self.width, 16, 16, 8,9)

        self.conv2 = SpectralConv2d(4*self.width, 8*self.width, 8, 8,4,5)
        
        self.conv2_5 = SpectralConv2d(8*self.width, 8*self.width, 8, 8,4,5)
        
        self.conv3 = SpectralConv2d(8*self.width, 4*self.width, 16, 16,4,5)

        self.conv4 = SpectralConv2d(8*self.width, 2*self.width, 32, 32,8,9)

        self.conv5 = SpectralConv2d(4*self.width, self.width, 64, 64,16,17) # will be reshaped

        self.w0 = pointwise_op(self.width,2*self.width,32, 32) #
        
        self.w1 = pointwise_op(2*self.width, 4*self.width, 16, 16) #
        
        self.w2 = pointwise_op(4*self.width, 8*self.width, 8, 8) #
        
        self.w2_5 = pointwise_op(8*self.width, 8*self.width, 8, 8) #

        self.w3 = pointwise_op(8*self.width, 4*self.width, 16, 16) #
        
        self.w4 = pointwise_op(8*self.width, 2*self.width, 32, 32)
        
        self.w5 = pointwise_op(4*self.width, self.width, 64, 64) # will be reshaped

        self.fc1 = nn.Linear(2*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        
        x_fc0 = self.fc0(x)
        x_fc0 = F.selu(x_fc0,inplace=True)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        x_fc0 = F.pad(x_fc0, [self.padding,self.padding, self.padding,self.padding])
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]
        '''
        Please note that, in this implementation of UNO, at each block the grid size is halved.
        So, at the each block I am deviding the grid size by the powers of 2.
        '''
        x1_c0 = self.conv0(x_fc0, D1//2,D2//2)
        x2_c0 = self.w0(x_fc0, D1//2,D2//2)
        x_c0 = x1_c0 + x2_c0
        x_c0 = F.selu(x_c0,inplace=True)
        #print(x.shape)

        x1_c1 = self.conv1(x_c0,D1//4,D2//4)
        x2_c1 = self.w1(x_c0,D1//4,D2//4)
        x_c1 = x1_c1 + x2_c1
        x_c1 = F.selu(x_c1,inplace=True)
        #print(x.shape)

        x1_c2 = self.conv2(x_c1,D1//8,D2//8)
        x2_c2 = self.w2(x_c1,D1//8,D2//8)
        x_c2 = x1_c2 + x2_c2
        x_c2 = F.selu(x_c2,inplace=True)
        #print(x.shape)
        
        x1_c2_5 = self.conv2_5(x_c2,D1//8,D2//8)
        x2_c2_5 = self.w2_5(x_c2,D1//8,D2//8)
        x_c2_5 = x1_c2_5 + x2_c2_5
        x_c2_5 = F.selu(x_c2_5,inplace=True)
        #print(x.shape)

        x1_c3 = self.conv3(x_c2_5,D1//4,D2//4)
        x2_c3 = self.w3(x_c2_5,D1//4,D2//4)
        x_c3 = x1_c3 + x2_c3
        x_c3 = F.selu(x_c3,inplace=True)
        x_c3 = torch.cat([x_c3, x_c1], dim=1)

        x1_c4 = self.conv4(x_c3,D1//2,D2//2)
        x2_c4 = self.w4(x_c3,D1//2,D2//2)
        x_c4 = x1_c4 + x2_c4
        x_c4 = F.selu(x_c4,inplace=True)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)

        x1_c5 = self.conv5(x_c4,D1,D2)
        x2_c5 = self.w5(x_c4,D1,D2)
        x_c5 = x1_c5 + x2_c5
        x_c5 = F.selu(x_c5,inplace=True)
        #print(x.shape)
        x_c5 = torch.cat([x_c5, x_fc0], dim=1)

        x_c5 = x_c5[..., self.padding:-self.padding, self.padding:-self.padding]

        x_c5 = x_c5.permute(0, 2, 3, 1)
        x_fc1 = self.fc1(x_c5)
        x_fc1 = F.selu(x_fc1,inplace=True)

        x_out = self.fc2(x_fc1)
        
        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)