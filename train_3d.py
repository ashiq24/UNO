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



def train_model_3d(model,train_loader,val_loader,test_loader,ntrain,nval,ntest,weight_path,\
    T_f=10,step=1,batch_size=20,epochs=150,learning_rate= 0.0001,\
    scheduler_step= 50,scheduler_gamma= 0.5,device = 'cuda',x_normalizer = None, y_normalizer = None, weight_dec = 1e-3):
    
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_dec,amsgrad = False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    Min_error_t = 100000.000
    myloss = LpLoss(size_average=False)

    for ep in range(epochs):
        t1 = default_timer()
        train_l2_step = 0
        model.train()
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            batch_size = x.shape[0]
            S = x.shape[1]
            optimizer.zero_grad()
            out = model(x).view(batch_size, S, S, T_f)

            temp_step_loss = 0
            with torch.no_grad():
                for time in range(T_f):
                    k,l = out[...,time],y[...,time]
                    temp_step_loss+=myloss(k.view(batch_size, -1), l.view(batch_size, -1))
                train_l2_step += temp_step_loss.item()

            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward()

            optimizer.step()
            
            del x,y,out,l2
            gc.collect()
            
        scheduler.step()
        gc.collect()
        torch.cuda.empty_cache()
        
        train_l2_step /= ntrain*T_f
        if ep%2 == 1:
            t2 = default_timer()
            print("epochs",ep, "time",t2 - t1, "train_loss ",train_l2_step)
            continue
        
        model.eval()
        val_l2_step = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                batch_size = x.shape[0]
                out = model(x).view(batch_size, S, S, T_f)
                #out = y_normalizer.decode(out)

                temp_step_loss = 0
                for time in range(T_f):
                    k,l = out[...,time],y[...,time]
                    temp_step_loss+=myloss(k.view(batch_size, -1), l.view(batch_size, -1))

                val_l2_step+=temp_step_loss.item()
                
                del x,y,out

        gc.collect()
        val_l2_step /= nval*T_f

        t2 = default_timer()
        print("epochs",ep, "time",t2 - t1, "train_loss ",train_l2_step,"Val_loss  ", val_l2_step)
        torch.cuda.empty_cache()
        if Min_error_t > val_l2_step:
            torch.save(model.state_dict(), weight_path)
            print("model saved", Min_error_t - val_l2_step)
            Min_error_t = val_l2_step

    print("Traning Ended")
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    test_l2_step = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            batch_size = x.shape[0]
            out = model(x).view(batch_size, S, S, T_f)

            temp_step_loss = 0
            for time in range(T_f):
                k,l = out[...,time],y[...,time]
                temp_step_loss+=myloss(k.view(batch_size, -1), l.view(batch_size, -1))

            test_l2_step+=temp_step_loss.item()
            
            del x,y,out

    gc.collect()
    test_l2_step /= ntest*T_f

    print("*** Test error: ", test_l2_step)
    