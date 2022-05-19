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



def train_model(model,train_loader, val_loader, test_loader,ntrain,nval,ntest,s,wieght_path,T_f=10,step=1,batch_size=20,epochs=150,learning_rate= 0.0001,\
scheduler_step= 50,scheduler_gamma= 0.5,device = 'cuda', weight_decay = 1e-3):
    
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay,amsgrad = False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    best_error = 100000.0
    myloss = LpLoss(size_average=False)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            batch_size = x.shape[0]
            optimizer.zero_grad()
            out = model(x).reshape(batch_size, s, s)

            loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()
            del x,y,out,loss
            gc.collect()
        torch.cuda.empty_cache()

        scheduler.step()

        model.eval()
        val_l2 = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                batch_size = x.shape[0]
                out = model(x).reshape(batch_size, s, s)

                val_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

        train_l2/= ntrain
        val_l2 /= nval

        t2 = default_timer()
        if best_error > val_l2:
            print("..Saving Model..", best_error - val_l2)
            best_error = val_l2
            torch.save(model.state_dict(), wieght_path)
        print(ep, t2-t1, train_l2, val_l2)
        
    model.load_state_dict(torch.load(wieght_path))
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            batch_size = x.shape[0]
            out = model(x).reshape(batch_size, s, s)

            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2-t1, "Test Error", test_l2)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    