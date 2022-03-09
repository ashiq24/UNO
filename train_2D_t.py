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



def train_model(model,train_loader, test_loader,ntrain,ntest,T_f=10,step=1,batch_size=20,epochs=150,learning_rate= 0.0001,\
scheduler_step= 50,scheduler_gamma= 0.5,device = 'cuda',x_normalizer = None, y_normalizer = None, weight_dec = 1e-3):
    
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_dec,amsgrad = False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    myloss = LpLoss(size_average=False)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        for xx, yy in train_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            xx = xx #+ 0.0005*torch.randn(xx.shape).cuda()
            
            '''if random.random()< 0.25:
                xx = torch.transpose(xx,1,2)
                yy = torch.transpose(yy,1,2)'''
            
            for t in range(0, T_f, step):
                y = yy[..., t:t + step]
                im = model(xx)
                
                #denormalizing
                #im = y_normalizer.decode(im)
                
                
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            train_l2_step += loss.item()
            l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
            train_l2_full += l2_full.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            del xx,yy,pred
            
        gc.collect()
        
        if ep%2 == 1:
            t2 = default_timer()
            print("epochs",ep, "time",t2 - t1, "train_step",train_l2_step / ntrain / (T_f / step),"train_full", train_l2_full / ntrain)
            continue
        test_l2_step = 0
        test_l2_full = 0
        with torch.no_grad():
            for xx, yy in test_loader:
                xx = xx.to(device)
                yy = yy.to(device)
                loss = 0
                for t in range(0, T_f, step):
                    y = yy[..., t:t + step]
                    im = model(xx)
                    
                    #denormalized version
                    #im1 = y_normalizer.decode(im)
                    
                    loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                    xx = torch.cat((xx[..., step:], im), dim=-1)

                test_l2_step += loss.item()
                test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
                del xx,yy,pred

        t2 = default_timer()
        scheduler.step()
        print("epochs",ep, "time",t2 - t1, "train_step",train_l2_step / ntrain / (T_f / step),"train_full", train_l2_full / ntrain,"Test_step", test_l2_step / ntest / (T_f / step),
            "test_full",test_l2_full / ntest)