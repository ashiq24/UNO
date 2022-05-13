
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



class SpectralConv3d_UNO(nn.Module):
    def __init__(self, in_channels, out_channels,D1,D2,D3, modes1=None, modes2=None, modes3=None):
        super(SpectralConv3d_UNO, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        D1,D2,D3 are output dimensions (x,y,t)
        modes1,modes2,modes3 = Number of fourier coefficinets to consider along each spectral dimesion   
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
            self.modes1 = D1 #Will take the maximum number of possiblel modes for given output dimension
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
        D1,D2,D3 are the output dimensions (x,y,t)
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
    def __init__(self, in_channel, out_channel,dim1, dim2,dim3):
        super(pointwise_op_3D,self).__init__()
        self.conv = nn.Conv3d(int(in_channel), int(out_channel), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.dim3 = int(dim3)

    def forward(self,x, dim1 = None, dim2 = None, dim3 = None):
        """
        dim1,dim2,dim3 are the output dimensions (x,y,t)
        """
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
            dim3 = self.dim3
        x_out = self.conv(x)
        x_out = torch.nn.functional.interpolate(x_out, size = (dim1, dim2,dim3),mode = 'trilinear',align_corners=True)
        return x_out

class OperatorBlock_3D(nn.Module,):
    def __init__(self, in_channel, out_channel,dim1, dim2,dim3,modes1,modes2,modes3, Normalize = True,Non_Lin = True):
        super(OperatorBlock_3D,self).__init__()
        self.conv = SpectralConv3d_UNO(in_channel, out_channel, dim1,dim2,dim3,modes1,modes2,modes3)
        self.w = pointwise_op_3D(in_channel, out_channel, dim1,dim2,dim3)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm3d(int(out_channel),affine=True)


    def forward(self,x, dim1 = None, dim2 = None, dim3 = None):

        x1_out = self.conv(x,dim1,dim2,dim3)
        x2_out = self.w(x,dim1,dim2,dim3)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out


class Uno3D_T40(nn.Module):
    """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by  self.fc, self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps (u(1), ..., u(10)).
        input shape: (batchsize, x=64, y=64, t=10, c=1)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """
    def __init__(self, in_width, width,pad = 1, factor = 1, pad_both = False):
        super(Uno3D_T40, self).__init__()

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  
        self.pad_both = pad_both

        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width)  
        
        self.conv0 = OperatorBlock_3D(self.width, 2*factor*self.width,48, 48, 10, 24,24, 5)
        
        self.conv1 = OperatorBlock_3D(2*factor*self.width, 4*factor*self.width, 32, 32,10,  16,16,5)
        
        self.conv2 = OperatorBlock_3D(4*factor*self.width, 8*factor*self.width, 16, 16, 16,  8,8,5)
        
        self.conv3 = OperatorBlock_3D(8*factor*self.width, 16*factor*self.width, 8, 8, 16,  4,4,8)
        
        self.conv4 = OperatorBlock_3D(16*factor*self.width, 16*factor*self.width, 8, 8, 16,  4,4,8)
        
        self.conv5 = OperatorBlock_3D(16*factor*self.width, 8*factor*self.width, 16, 16, 16,  4,4,8) 
        
        self.conv6 = OperatorBlock_3D(8*factor*self.width, 4*factor*self.width, 32, 32, 24,  8,8,8)
        
        self.conv7 = OperatorBlock_3D(8*factor*self.width, 2*factor*self.width, 48, 48, 32,  16,16,12)
        
        self.conv8 = OperatorBlock_3D(4*factor*self.width, 2*self.width, 64, 64, 40,  24,24, 16) # will be reshaped

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
    input: the solution of the first 10 timesteps (u(1), ..., u(10)).
    input shape: (batchsize, x=64, y=64, t=10, c=1)
    output: the solution of the next 20 timesteps
    output shape: (batchsize, x=64, y=64, t=20, c=1)

    """
    def __init__(self, in_width, width,pad = 2, factor = 1, pad_both = False):
        super(Uno3D_T20, self).__init__()

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  
        self.pad_both = pad_both

        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width)  
        
        self.conv0 = OperatorBlock_3D(self.width, 2*factor*self.width,48, 48, 10, 24,24, 5)
        
        self.conv1 = OperatorBlock_3D(2*factor*self.width, 4*factor*self.width, 32, 32,10,  16,16,5)
        
        self.conv2 = OperatorBlock_3D(4*factor*self.width, 8*factor*self.width, 16, 16, 12,  8,8,5)
        
        self.conv3 = OperatorBlock_3D(8*factor*self.width, 16*factor*self.width, 8, 8, 12,  4,4,6)
        
        self.conv4 = OperatorBlock_3D(16*factor*self.width, 16*factor*self.width, 8, 8, 16,  4,4,6)
        
        self.conv5 = OperatorBlock_3D(16*factor*self.width, 8*factor*self.width, 16, 16, 16,  4,4,8) 
        
        self.conv6 = OperatorBlock_3D(8*factor*self.width, 4*factor*self.width, 32, 32, 18,  8,8,8)
        
        self.conv7 = OperatorBlock_3D(8*factor*self.width, 2*factor*self.width, 48, 48, 20,  16,16,8)
        
        self.conv8 = OperatorBlock_3D(4*factor*self.width, 2*self.width, 64, 64, 20,  24,24, 8) # will be reshaped

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
    input: the solution of the first 10 timesteps (u(1), ..., u(10)).
    input shape: (batchsize, x=64, y=64, t=10, c=1)
    output: the solution of the next 10 timesteps
    output shape: (batchsize, x=64, y=64, t=10, c=1)

    """
    def __init__(self, in_width, width,pad = 2, factor = 1, pad_both = False):
        super(Uno3D_T10, self).__init__()

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  
        self.pad_both = pad_both

        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width)  
        
        self.conv0 = OperatorBlock_3D(self.width, 2*factor*self.width,48, 48, 10, 24,24, 5)
        
        self.conv1 = OperatorBlock_3D(2*factor*self.width, 4*factor*self.width, 32, 32,10,  16,16,4)
        
        self.conv2 = OperatorBlock_3D(4*factor*self.width, 8*factor*self.width, 16, 16, 8,  8,8,4)
        
        self.conv3 = OperatorBlock_3D(8*factor*self.width, 16*factor*self.width, 8, 8, 8,  4,4,4)
        
        self.conv4 = OperatorBlock_3D(16*factor*self.width, 16*factor*self.width, 8, 8, 8,  4,4,4)
        
        self.conv5 = OperatorBlock_3D(16*factor*self.width, 8*factor*self.width, 16, 16, 8,  4,4,4) 
        
        self.conv6 = OperatorBlock_3D(8*factor*self.width, 4*factor*self.width, 32, 32, 8,  8,8,4)
        
        self.conv7 = OperatorBlock_3D(8*factor*self.width, 2*factor*self.width, 48, 48, 10,  16,16,4)
        
        self.conv8 = OperatorBlock_3D(4*factor*self.width, 2*self.width, 64, 64, 10,  24,24, 5) # will be reshaped

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
    input: the solution of the first 6 timesteps (u(1), ..., u(10)).
    input shape: (batchsize, x=64, y=64, t=6, c=1)
    output: the solution of the next 9 timesteps
    output shape: (batchsize, x=64, y=64, t=9, c=1)

    """
    def __init__(self, in_width, width,pad = 2, factor = 1, pad_both = False):
        super(Uno3D_T9, self).__init__()

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  
        self.pad_both = pad_both

        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width)  
        
        self.conv0 = OperatorBlock_3D(self.width, 2*factor*self.width,48, 48, 6, 24,24, 3)
        
        self.conv1 = OperatorBlock_3D(2*factor*self.width, 4*factor*self.width, 32, 32,6,  16,16,3)
        
        self.conv2 = OperatorBlock_3D(4*factor*self.width, 8*factor*self.width, 16, 16, 6,  8,8,3)
        
        self.conv3 = OperatorBlock_3D(8*factor*self.width, 16*factor*self.width, 8, 8, 8,  4,4,3)
        
        self.conv4 = OperatorBlock_3D(16*factor*self.width, 16*factor*self.width, 8, 8, 8,  4,4,4)
        
        self.conv5 = OperatorBlock_3D(16*factor*self.width, 8*factor*self.width, 16, 16, 8,  4,4,4) 
        
        self.conv6 = OperatorBlock_3D(8*factor*self.width, 4*factor*self.width, 32, 32, 8,  8,8,4)
        
        self.conv7 = OperatorBlock_3D(8*factor*self.width, 2*factor*self.width, 48, 48, 9,  16,16,4)
        
        self.conv8 = OperatorBlock_3D(4*factor*self.width, 2*self.width, 64, 64, 9,  24,24, 4) # will be reshaped

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