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



def train_model(model,train_loader,val_loader,test_loader,ntrain,nval,ntest,weight_path,T_f=10,step=1,batch_size=20,epochs=150,learning_rate= 0.0001,\
scheduler_step= 100,scheduler_gamma= 0.5,device = 'cuda', weight_decay = 1e-3):
    
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay,amsgrad = False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    Min_error_t = 100000.000
    myloss = LpLoss(size_average=False)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2_step = 0
        for xx, yy in train_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            batch_size = yy.shape[0]
            
            
            for t in range(0, T_f, step):
                y = yy[..., t:t + step]
                im = model(xx)    
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)
            with torch.no_grad():
                train_l2_step += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            del xx,yy,pred
            
        gc.collect()
        
        if ep%2 == 1:
            t2 = default_timer()
            print("epochs",ep, "time",t2 - t1, "train_loss",train_l2_step / ntrain / (T_f / step))
            continue
        
        model.eval()
        val_l2_step = 0
        with torch.no_grad():
            for xx, yy in val_loader:
                xx = xx.to(device)
                yy = yy.to(device)
                loss = 0
                batch_size = yy.shape[0]
                for t in range(0, T_f, step):
                    y = yy[..., t:t + step]
                    im = model(xx)
                    
                    loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                    xx = torch.cat((xx[..., step:], im), dim=-1)

                val_l2_step += loss.item()
                del xx,yy,pred

        t2 = default_timer()
        scheduler.step()
        if Min_error_t > val_l2_step / nval / (T_f / step):
            torch.save(model.state_dict(), weight_path)
            print("model saved", Min_error_t - val_l2_step / nval / (T_f / step))
            Min_error_t = val_l2_step / nval / (T_f / step)
            
        print("epochs",ep, "time",t2 - t1, "train_loss ",train_l2_step / ntrain / (T_f / step),"val_loss", val_l2_step / nval / (T_f / step))
    
    print("Traning Ended")
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    test_l2_step = 0
    test_l2 = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            loss = 0
            batch_size = yy.shape[0]
            for t in range(0, T_f, step):
                y = yy[..., t:t + step]
                im = model(xx)

                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2 += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()


            del xx,yy,pred

    t2 = default_timer()
    scheduler.step()
    print("Test set Evaluation ","Test_loss", test_l2_step / ntest / (T_f / step), test_l2 / ntest)
