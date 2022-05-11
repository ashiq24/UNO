import torch
from data import *
from model3D import FNO3d,Uno3D_T40
from train_3d import train_model_3d
from utilities3 import *
from torchsummary import summary
from test import plot_aRun
torch.cuda.set_device(2)

S = 64 #resolution (Grid size) of training data
T_in = 1 #Number of input time steps (e.g. vorticity of first 10 time steps of a fuild flow foir Navier Stoks equation)
T_f = 40  ##Number of output time steps (e.g. vorticity of next 40 time steps of a fuild flow foir Navier Stoks equation)
ntrain = 100 # Numbe of training samples
ntest = 20 # Number of test sample
batch_size = 20 # batch size 
width = 8 # In UNO, at the begining we lift the input to a higher dimension channel space. 'Width' is the dimension of that space
inwidth = T_in+3 # the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y). The total number of input channel
# please note that +3 is for positional embedding (x,y,t)
epochs = 2 # Number of epochs to train
print(T_in,T_f)
train_a = torch.rand((100,64,64,T_in))
train_u = torch.rand((100,64,64,T_f))
val_a = torch.rand((20,64,64,T_in))
val_u = torch.rand((20,64,64,T_f))


#train_a = train_a.reshape(ntrain,S,S,1,T_in).repeat([1,1,1,T_f,1])
#val_a = val_a.reshape(ntest,S,S,1,T_in).repeat([1,1,1,T_f,1])

train_a = train_a.reshape(ntrain,S,S,T_in,1)
val_a = val_a.reshape(ntest,S,S,T_in,1)


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_a, val_u), batch_size=batch_size, shuffle=False)

model = Uno3D_T40(4,8).cuda()
summary(model, (64, 64,T_in,1))
train_model_3d(model,train_loader,val_loader, ntrain,ntest,\
            T_f=T_f,batch_size=batch_size,epochs=epochs,learning_rate= 0.001,\
            x_normalizer = None, y_normalizer = None, scheduler_step= 50,scheduler_gamma= 0.5,weight_dec = 1e-4)