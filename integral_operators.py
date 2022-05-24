import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d_Uno(nn.Module):
    def __init__(self, in_channels, out_channels, dim1,modes1 = None):
        super(SpectralConv1d_Uno, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim1 = dim1 #output dimensions
        if modes1 is not None:
            self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        else:
            self.modes1 = dim1//2

        self.scale = (1 / (2*in_channels))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, dim1 = None):
        if dim1 is not None:
            self.dim1 = dim1
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  self.dim1//2 + 1 , dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=self.dim1)
        return x

class pointwise_op_1D(nn.Module):
    def __init__(self, in_channel, out_channel,dim1):
        super(pointwise_op_1D,self).__init__()
        self.conv = nn.Conv1d(int(in_channel), int(out_channel), 1)
        self.dim1 = int(dim1)

    def forward(self,x, dim1 = None):
        if dim1 is None:
            dim1 = self.dim1
        x_out = self.conv(x)
        x_out = torch.nn.functional.interpolate(x_out, size = dim1,mode = 'linear',align_corners=True)
        return x_out



class SpectralConv2d_Uno(nn.Module):
    def __init__(self, in_codim, out_codim, dim1, dim2,modes1 = None, modes2 = None):
        super(SpectralConv2d_Uno, self).__init__()

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

class pointwise_op_2D(nn.Module):
    """ 
    dim1 = Default output grid size along x (or 1st dimension) 
    dim2 = Default output grid size along y ( or 2nd dimension)
    in_codim = Input co-domian dimension
    out_codim = output co-domain dimension
    """
    def __init__(self, in_codim, out_codim,dim1, dim2):
        super(pointwise_op_2D,self).__init__()
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

class OperatorBlock_2D(nn.Module,):
    def __init__(self, in_channel, out_channel,dim1, dim2,modes1,modes2, Normalize = False,Non_Lin = True):
        super(OperatorBlock_3D,self).__init__()
        self.conv = SpectralConv2d_Uno(in_channel, out_channel, dim1,dim2,modes1,modes2)
        self.w = pointwise_op_2D(in_channel, out_channel, dim1,dim2)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm2d(int(out_channel),affine=True)


    def forward(self,x, dim1 = None, dim2 = None):

        x1_out = self.conv(x,dim1,dim2)
        x2_out = self.w(x,dim1,dim2)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out


class SpectralConv3d_Uno(nn.Module):
    def __init__(self, in_codim, out_codim,D1,D2,D3, modes1=None, modes2=None, modes3=None):
        super(SpectralConv3d_Uno, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        D1 = Default output grid size along x (or 1st dimension) 
        D2 = Default output grid size along y ( or 2nd dimension)
        D3 = Default output grid size along time t ( or 3rd dimension)
        Ratio of grid size of the input and output grid size (D1,D2,D3) implecitely 
        set the expansion or contraction farctor along each dimension.
        modes1, modes2, modes3 = Number of fourier modes to consider for the ontegral operator
                                Number of modes must be compatibale with the input grid size 
                                and desired output grid size.
                                i.e., modes1 <= min( dim1/2, input_dim1/2). 
                                Here input_dim1 is the grid size along x axis (or first dimension) of the input.
                                Other modes also have the same constrain.
        in_codim = Input co-domian dimension
        out_codim = output co-domain dimension   
        """
        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.d1 = D1
        self.d2 = D2
        self.d3 = D3
        if modes1 is not None:
            self.modes1 = modes1 
            self.modes2 = modes2
            self.modes3 = modes3 
        else:
            self.modes1 = D1 
            self.modes2 = D2
            self.modes3 = D3//2+1

        self.scale = (1 / (2*in_codim))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):

        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x, D1 = None,D2=None,D3=None):
        """
        D1,D2,D3 are the output grid size along (x,y,t)
        """
        if D1 is not None:
            self.d1 = D1
            self.d2 = D2
            self.d3 = D3   

        batchsize = x.shape[0]

        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

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
    def __init__(self, in_codim, out_codim,dim1, dim2,dim3):
        super(pointwise_op_3D,self).__init__()
        self.conv = nn.Conv3d(int(in_codim), int(out_codim), 1)
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
        self.conv = SpectralConv3d_Uno(in_channel, out_channel, dim1,dim2,dim3,modes1,modes2,modes3)
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

