# Codes for section: Results on Navier Stocks Equation (2D)
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
    def __init__(self, in_codim, out_codim, dim1, dim2,modes1 = None, modes2 = None):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        dim1 = Default output grid size along x (or 1st dimension) 
        dim2 = Default output grid size along y ( or 2nd dimension)
        Ratio of grid size of the input and output grid size (dim1,dim2) implecitely 
        set the expansion or contraction farctor along each dimension.
        modes1, modes2 = Number of fourier modes to consider for the ontegral operator
                        Number of modes must be compatibale with the input grid size 
                        and desired output grid size.
                        i.e., modes1 <= min( dim1/2, input_dim1/2). 
                        Here input_dim1 is the grid size along x axis (or first dimension) of the input.
                        Other modes also the have same constrain.
        in_codim = Input co-domian dimension
        out_codim = output co-domain dimension
        """

        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1 
        self.dim2 = dim2
        if modes1 is not None:
            self.modes1 = modes1 
            self.modes2 = modes2
        else:
            self.modes1 = dim1//2-1 
            self.modes2 = dim2//2 
        self.scale = (1 / (2*in_codim))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * (torch.randn(in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat)))
        self.weights2 = nn.Parameter(self.scale * (torch.randn(in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat)))

    # Complex multiplication
    def compl_mul2d(self, input, weights):

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
    """ 
    dim1 = Default output grid size along x (or 1st dimension) 
    dim2 = Default output grid size along y ( or 2nd dimension)
    in_codim = Input co-domian dimension
    out_codim = output co-domain dimension
    """
    def __init__(self, in_codim, out_codim,dim1, dim2):
        super(pointwise_op,self).__init__()
        self.conv = nn.Conv2d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)

    def forward(self,x, dim1 = None, dim2 = None):
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
        x_out = self.conv(x)
        x_out = torch.nn.functional.interpolate(x_out, size = (dim1, dim2),mode = 'bicubic',align_corners=True)
        return x_out

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
        input shape: (batchsize, x=S, y=S, t=10, c=1)
        output: the solution of the next timesteps
        output shape: (batchsize, x=S, y=S, t=1, c=1)
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

        self.conv0 = SpectralConv2d(self.width, 2*factor*self.width,32, 32, 16, 17)

        self.conv1 = SpectralConv2d(2*factor*self.width, 4*factor*self.width, 16, 16, 8,9)

        self.conv2 = SpectralConv2d(4*factor*self.width, 8*factor*self.width, 8, 8,4,5)
        
        self.conv3 = SpectralConv2d(8*factor*self.width, 8*factor*self.width, 8, 8,4,5)
        
        self.conv4 = SpectralConv2d(8*factor*self.width, 4*factor*self.width, 16, 16,4,5)

        self.conv5 = SpectralConv2d(8*factor*self.width, 2*factor*self.width, 32, 32,8,9)

        self.conv6 = SpectralConv2d(4*factor*self.width, self.width, 64, 64,16,17) # will be reshaped

        self.w0 = pointwise_op(self.width,2*factor*self.width,32, 32) #
        
        self.w1 = pointwise_op(2*factor*self.width, 4*factor*self.width, 16, 16) #
        
        self.w2 = pointwise_op(4*factor*self.width, 8*factor*self.width, 8, 8) #
        
        self.w3 = pointwise_op(8*factor*self.width, 8*factor*self.width, 8, 8) #

        self.w4 = pointwise_op(8*factor*self.width, 4*factor*self.width, 16, 16) #
        
        self.w5 = pointwise_op(8*factor*self.width, 2*factor*self.width, 32, 32)
        
        self.w6 = pointwise_op(4*factor*self.width, self.width, 64, 64) # will be reshaped

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
        
        x1_c3 = self.conv3(x_c2,D1//8,D2//8)
        x2_c3 = self.w3(x_c2,D1//8,D2//8)
        x_c3 = x1_c3 + x2_c3
        x_c3 = F.gelu(x_c3)
        #print(x.shape)

        x1_c4 = self.conv4(x_c3 ,D1//4,D2//4)
        x2_c4 = self.w4(x_c3 ,D1//4,D2//4)
        x_c4 = x1_c4 + x2_c4
        x_c4 = F.gelu(x_c4)
        x_c4 = torch.cat([x_c4, x_c1], dim=1)

        x1_c5 = self.conv5(x_c4 ,D1//2,D2//2)
        x2_c5 = self.w5(x_c4 ,D1//2,D2//2)
        x_c5 = x1_c5 + x2_c5
        x_c5 = F.gelu(x_c5)
        x_c5 = torch.cat([x_c5, x_c0], dim=1)

        x1_c6 = self.conv6(x_c5,D1,D2)
        x2_c6 = self.w6(x_c5,D1,D2)
        x_c6 = x1_c6 + x2_c6
        x_c6 = F.gelu(x_c6)
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

#####
#  The following models UNO has same sets of parameter 
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

        self.conv0 = SpectralConv2d(self.width, 2*factor*self.width,48, 48, 24, 24)

        self.conv1 = SpectralConv2d(2*factor*self.width, 4*factor*self.width, 32, 32, 16,16)

        self.conv2 = SpectralConv2d(4*factor*self.width, 8*factor*self.width, 16, 16,8,8)
        
        self.conv3 = SpectralConv2d(8*factor*self.width, 8*factor*self.width, 16, 16,8,8)
        
        self.conv4 = SpectralConv2d(8*factor*self.width, 4*factor*self.width, 32, 32,8,8)

        self.conv5 = SpectralConv2d(8*factor*self.width, 2*factor*self.width, 48, 48,16,16)

        self.conv6 = SpectralConv2d(4*factor*self.width, self.width, 64, 64,24,24) # will be reshaped

        self.w0 = pointwise_op(self.width,2*factor*self.width,48, 48) #
        
        self.w1 = pointwise_op(2*factor*self.width, 4*factor*self.width, 32, 32) #
        
        self.w2 = pointwise_op(4*factor*self.width, 8*factor*self.width, 16, 16) #
        
        self.w3 = pointwise_op(8*factor*self.width, 8*factor*self.width, 16, 16)
        
        self.w4 = pointwise_op(8*factor*self.width, 4*factor*self.width, 32, 32) #
        
        self.w5 = pointwise_op(8*factor*self.width, 2*factor*self.width, 48, 48)
        
        self.w6 = pointwise_op(4*factor*self.width, self.width, 64, 64) # will be reshaped

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

        self.conv0 = SpectralConv2d(self.width, 2*factor*self.width,64, 64, 32, 33)

        self.conv1 = SpectralConv2d(2*factor*self.width, 4*factor*self.width, 16, 16, 8,9)

        self.conv2 = SpectralConv2d(4*factor*self.width, 8*factor*self.width, 8, 8,4,5)
        
        self.conv3 = SpectralConv2d(8*factor*self.width, 8*factor*self.width, 8, 8,4,5)
        
        self.conv4 = SpectralConv2d(8*factor*self.width, 4*factor*self.width, 16, 16,4,5)

        self.conv5 = SpectralConv2d(8*factor*self.width, 2*factor*self.width, 64, 64,8,9)

        self.conv6 = SpectralConv2d(4*factor*self.width, self.width, 256, 256,32,32) # will be reshaped

        self.w0 = pointwise_op(self.width,2*factor*self.width,64, 64) #
        
        self.w1 = pointwise_op(2*factor*self.width, 4*factor*self.width, 16, 16) #
        
        self.w2 = pointwise_op(4*factor*self.width, 8*factor*self.width, 8, 8) #
        
        self.w3 = pointwise_op(8*factor*self.width, 8*factor*self.width, 8, 8) #

        self.w4 = pointwise_op(8*factor*self.width, 4*factor*self.width, 16, 16) #
        
        self.w5 = pointwise_op(8*factor*self.width, 2*factor*self.width, 64, 64)
        
        self.w6 = pointwise_op(4*factor*self.width, self.width, 256, 256) # will be reshaped

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

        x1_c0 = self.conv0(x_fc0,D1//4,D2//4)
        x2_c0 = self.w0(x_fc0,D1//4,D2//4)
        x_c0 = x1_c0 + x2_c0
        x_c0 = F.gelu(x_c0)
        #print(x.shape)

        x1_c1 = self.conv1(x_c0,D1//16,D2//16)
        x2_c1 = self.w1(x_c0,D1//16,D2//16)
        x_c1 = x1_c1 + x2_c1
        x_c1 = F.gelu(x_c1)
        #print(x.shape)

        x1_c2 = self.conv2(x_c1,D1//32,D2//32)
        x2_c2 = self.w2(x_c1,D1//32,D2//32)
        x_c2 = x1_c2 + x2_c2
        x_c2 = F.gelu(x_c2)
        #print(x.shape)
        
        x1_c3 = self.conv3(x_c2,D1//32,D2//32)
        x2_c3 = self.w3(x_c2,D1//32,D2//32)
        x_c3 = x1_c3 + x2_c3
        x_c3 = F.gelu(x_c3)
        #print(x.shape)

        x1_c4 = self.conv4(x_c3 ,D1//16,D2//16)
        x2_c4 = self.w4(x_c3 ,D1//16,D2//16)
        x_c4 = x1_c4 + x2_c4
        x_c4 = F.gelu(x_c4)
        x_c4 = torch.cat([x_c4, x_c1], dim=1)

        x1_c5 = self.conv5(x_c4 ,D1//4,D2//4)
        x2_c5 = self.w5(x_c4 ,D1//4,D2//4)
        x_c5 = x1_c5 + x2_c5
        x_c5 = F.gelu(x_c5)
        x_c5 = torch.cat([x_c5, x_c0], dim=1)

        x1_c6 = self.conv6(x_c5,D1,D2)
        x2_c6 = self.w6(x_c5,D1,D2)
        x_c6 = x1_c6 + x2_c6
        x_c6 = F.gelu(x_c6)
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