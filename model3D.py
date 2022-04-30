
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)


################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(13, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        '''self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)'''

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


################################################################
# Code of UNO3D starts 
# Pointwise and Fourier Layer
################################################################


class SpectralConv3d_UNO(nn.Module):
    def __init__(self, in_channels, out_channels,D1,D2,D3, modes1=None, modes2=None, modes3=None):
        super(SpectralConv3d_UNO, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        D1,D2,D3 are output resolution (x,y,t)
        modes1,modes2,modes3 = Number of fourier coefficinets to consider along each spectral resesion   
        """
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d1 = D1
        self.d2 = D2
        self.d3 = D3
        if modes1 is not None:
            self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes2 = modes2
            self.modes3 = modes3 
        else:
            self.modes1 = D1 #Will take the maximum number of possiblel modes for given output resension
            self.modes2 = D2
            self.modes3 = D3//2+1

        self.scale = (1 / (2*in_channels))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x, D1 = None,D2=None,D3=None):
        """
        D1,D2,D3 are the output resolution (x,y,t)
        """
        if D1 is not None:
            self.d1 = D1
            self.d2 = D2
            self.d3 = D3   

        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, self.d1, self.d2, self.d3//2 + 1, dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(self.d1, self.d2, self.d3))
        return x

class pointwise_op_3D(nn.Module):
    def __init__(self, in_channel, out_channel,res1, res2,res3):
        super(pointwise_op_3D,self).__init__()
        self.conv = nn.Conv3d(int(in_channel), int(out_channel), 1)
        self.res1 = int(res1)
        self.res2 = int(res2)
        self.res3 = int(res3)

    def forward(self,x, res1 = None, res2 = None, res3 = None):
        """
        res1,res2,res3 are the output resolution (x,y,t)
        """
        if res1 is None:
            res1 = self.res1
            res2 = self.res2
            res3 = self.res3
        x_out = self.conv(x)
        x_out = torch.nn.functional.interpolate(x_out, size = (res1, res2,res3),mode = 'trilinear',align_corners=True)
        return x_out

class OperatorBlock_3D(nn.Module,):
    """
    To turn to normalization set Normalize = True
    To have linear operator set Non_Lin = False
    """
    def __init__(self, in_channel, out_channel,res1, res2,res3,modes1,modes2,modes3, Normalize = False,Non_Lin = True):
        super(OperatorBlock_3D,self).__init__()
        self.conv = SpectralConv3d_UNO(in_channel, out_channel, res1,res2,res3,modes1,modes2,modes3)
        self.w = pointwise_op_3D(in_channel, out_channel, res1,res2,res3)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm3d(out_channel,affine=True)


    def forward(self,x, res1 = None, res2 = None, res3 = None):

        x1_out = self.conv(x,res1,res2,res3)
        x2_out = self.w(x,res1,res2,res3)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out
"""
Following neural operator is desinged for predicting next 20 time steps from the input (Initial 10 time steps).
Lines for Normalization are commented out.
"""
class Uno3D_T20(nn.Module):
    def __init__(self, in_width, width,pad = 0, factor = 1):
        super(Uno3D_T20, self).__init__()

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  # pad the domain if input is non-periodic

        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)
        
        self.conv0 = OperatorBlock_3D(self.width, 2*self.width,32, 32, 16, 16,16, 8)
        
        self.conv1 = OperatorBlock_3D(2*self.width, 4*factor*self.width, 16, 16,8, 8,8,4)
        
        self.conv2 = OperatorBlock_3D(4*factor*self.width, 8*factor*self.width, 8, 8, 8, 4,4, 4) 
        
        self.conv3 = OperatorBlock_3D(8*factor*self.width, 16*factor*self.width, 4, 4, 4, 3,3, 3) 
        
        self.conv4 = OperatorBlock_3D(16*factor*self.width, 16*factor*self.width, 4, 4, 4, 3,3, 3)
        
        self.conv5 = OperatorBlock_3D(16*factor*self.width, 8*factor*self.width, 8, 8, 8, 3,3, 3)
        
        self.conv6 = OperatorBlock_3D(8*factor*self.width, 4*factor*self.width, 16, 16, 8, 4,4,3)
        
        self.conv7 = OperatorBlock_3D(8*factor*self.width, 2*self.width, 32, 32, 16, 8,8,3)
        
        self.conv8 = OperatorBlock_3D(4*self.width, self.width, 64, 64, 20, 16,16, 8) # will be reshaped

        self.fc1 = nn.Linear(2*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 1)
        
        #self.bn_fc_1 = torch.nn.InstanceNorm3d(self.width)
        #self.bn_fc0 = torch.nn.InstanceNorm3d(self.width)
        #self.bn_fc1 = torch.nn.InstanceNorm3d(4*self.width)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x_fc = self.fc(x)
        x_fc = F.gelu(x_fc)
        x_fc0 = self.fc0(x_fc)
        #x_fc0 = self.bn_fc0(x_fc0.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)
        
        x_fc0 = F.pad(x_fc0, [0,self.padding,0,0,0,0],mode ='constant')
        
        D1,D2,D3 = x_fc0.shape[-3],x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.conv0(x_fc0,D1//2,D2//2, 4*D3//5)
        x_c1 = self.conv1(x_c0,D1//4,D2//4, 2*D3//5)
        x_c2 = self.conv2(x_c1,D1//8,D2//8, 2*D3//5)
        
        x_c3 = self.conv3(x_c2,D1//16,D2//16, D3//5)
        x_c4 = self.conv4(x_c3,D1//16,D2//16, D3//5)

        x_c5 = self.conv5(x_c4,D1//8,D2//8, 2*D3//5)

        x_c6 = self.conv6(x_c5,D1//4,D2//4,2*D3//5)
        x_c6 = torch.cat([x_c6, x_c1], dim=1)

        x_c7 = self.conv7(x_c6,D1//2,D2//2,4*D3//5)
        x_c7 = torch.cat([x_c7, x_c0], dim=1)

        x_c8 = self.conv8(x_c7,D1,D2,D3)
        x_c8 = torch.cat([x_c8, x_fc0], dim=1)

        if self.padding!=0:
            x_c8 = x_c8[...,:-self.padding]

        x_c8 = x_c8.permute(0, 2, 3, 4, 1)

        x_fc1 = self.fc1(x_c8)
        #x_fc1 = self.bn_fc1(x_fc1.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
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