import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from functools import reduce
from functools import partial
import random
import gc
from timeit import default_timer
from utilities3 import *
from Adam import Adam



def train_model_3d(model,train_loader, test_loader,ntrain,ntest,T_f=10,step=1,batch_size=20,epochs=150,learning_rate= 0.0001,\
scheduler_step= 50,scheduler_gamma= 0.5,device = 'cuda',x_normalizer = None, y_normalizer = None, weight_dec = 1e-3):
    
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_dec,amsgrad = False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    myloss = LpLoss(size_average=False)

    for ep in range(epochs):
        t1 = default_timer()
        train_l2 = 0
        train_l2_step = 0
        model.train()
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            S = x.shape[1]
            optimizer.zero_grad()
            out = model(x).view(batch_size, S, S, T_f)

            #mse = F.mse_loss(out, y, reduction='mean')
            # mse.backward()

            #y = y_normalizer.decode(y)
            #out = y_normalizer.decode(out)
            temp_step_loss = 0
            with torch.no_grad():
                for time in range(T_f):
                    k,l = out[...,time],y[...,time]
                    temp_step_loss+=myloss(k.view(batch_size, -1), l.view(batch_size, -1))

            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward()

            optimizer.step()
            train_l2_step += temp_step_loss.item()
            train_l2 += l2.item()

            del x,y,out

        scheduler.step()
        gc.collect()

        train_l2_step /= ntrain*T_f
        train_l2 /= ntrain
        if ep%2 == 1:
            t2 = default_timer()
            print("epochs",ep, "time",t2 - t1, "train_step",train_l2_step,"train_full", train_l2,"Test_step", test_l2_step,
                "test_full",test_l2)
            continue
        
        model.eval()
        test_l2 = 0.0
        test_l2_step = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()

                out = model(x).view(batch_size, S, S, T_f)
                #out = y_normalizer.decode(out)
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

                temp_step_loss = 0
                for time in range(T_f):
                    k,l = out[...,time],y[...,time]
                    temp_step_loss+=myloss(k.view(batch_size, -1), l.view(batch_size, -1))

                test_l2_step+=temp_step_loss.item()


        test_l2 /= ntest
        test_l2_step /= ntest*T_f

        t2 = default_timer()
        print("epochs",ep, "time",t2 - t1, "train_step",train_l2_step,"train_full", train_l2,"Test_step", test_l2_step,
                "test_full",test_l2)
    