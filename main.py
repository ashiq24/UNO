import torch
from data import *
from UNO import FNO2d_UNO
from train_2D_t import train_model
from utilities3 import *
from torchsummary import summary
from test import plot_aRun


S = 64 #resolution (Grid size) of training data
T_in = 10 #Number of input time steps (e.g. vorticity of first 10 time steps of a fuild flow foir Navier Stoks equation)
T_f = 10  ##Number of output time steps (e.g. vorticity of next 10 time steps of a fuild flow foir Navier Stoks equation)
ntrain = 1000 # Numbe of training samples
ntest = 200 # Number of test sample
batch_size = 32 # batch size 
width = 16 # In UNO, at the begining we lift the input to a higher dimension channel space. 'Width' is the dimension of that space
inwidth = 12 # the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y). The total number of input channel
epochs = 2 # Number of epochs to train

train_a = torch.rand((1000,64,64,T_in))
train_u = torch.rand((1000,64,64,T_f))
val_a = torch.rand((200,64,64,T_in))
val_u = torch.rand((200,64,64,T_f))


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_a, val_u), batch_size=batch_size, shuffle=False)

model = FNO2d_UNO(inwidth,width).cuda()
summary(model, (64, 64,10))
train_model(model,train_loader,val_loader, ntrain,ntest,\
            T_f=T_f,batch_size=batch_size,epochs=epochs,learning_rate= 0.001,\
            x_normalizer = None, y_normalizer = None, scheduler_step= 50,scheduler_gamma= 0.5,weight_dec = 1e-4)