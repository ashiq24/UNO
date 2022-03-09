import numpy as np
import matplotlib.pyplot as plt
import torch

# Show images
def show_images(images, image_dim = (28,28)):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    for index, image in enumerate(images):
        plt.subplot(sqrtn, sqrtn, index+1)
        if image_dim !=(28,28):
            image = torch.squeeze(image)
            plt.imshow(image.permute(1, 2, 0))
        plt.axis('off')

def plot_aRun(model,xx,yy,T,step,device="cuda"):
    with torch.no_grad():
        xx = xx.to(device)
        yy = yy.to(device)
        idx = 0
        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            
            
            idx += 1
            plt.subplot(15, 3, idx)
            plt.imshow(torch.squeeze(im[0].cpu()))
            plt.axis('off')

            idx += 1 
            plt.subplot(15, 3, idx)
            plt.imshow(torch.squeeze(y[0].cpu()))
            plt.axis('off')
            
            idx += 1
            plt.subplot(15, 3, idx)
            plt.imshow(torch.squeeze(torch.abs(im[0]-y[0]).cpu()))
            plt.axis('off')
            xx = torch.cat((xx[..., step:], im), dim=-1)
        
        plt.show()
        

def visualize(model,device, test_loader, Type = "ANN", image_dim = (28,28)):
    with torch.no_grad():
        for data in test_loader:
            inputs = data[0]
            show_images(inputs,image_dim = image_dim)
            plt.show()

            inputs = inputs.to(device)
            if Type == "ANN":
                inputs = inputs.view(-1, 784)
            _, outputs,_ = model(inputs)
            show_images(outputs.cpu(),image_dim = image_dim)
            plt.show()
            del inputs,outputs
            break
def check_region(x,low,high):
    return x>=low and x<=high