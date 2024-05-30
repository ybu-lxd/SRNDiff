import os

import imageio
from matplotlib import pyplot as plt
import cv2
import einops
import numpy
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):

    # grid = torchvision.utils.make_grid(images, **kwargs)
    # print(grid.shape)
    # images = torch.cat([images[0,...],images[0,...],images[0,...],images[0,...]],dim=-1)
    # images = torch.cat([images[1, ...], images[1, ...], images[0, ...], images[0, ...]], dim=-1)
    # images = torch.cat([images[0, ...], images[0, ...], images[0, ...], images[0, ...]], dim=-1)
    # images = torch.cat([images[0, ...], images[0, ...], images[0, ...], images[0, ...]], dim=-1)

    imggrid=[]

    for i in range(images.shape[0]):
        # print(torch.cat([images[i,0,...],images[i,1,...],images[i,2,...],images[i,3,...]],dim=1).shape)
        imggrid.append(torch.cat([images[i,0,...],images[i,1,...],images[i,2,...],images[i,3,...]],dim=1))

    img = torch.stack(imggrid,dim=0)
    img = einops.rearrange(img,'c w h -> (c w) h')
    img = img.to('cpu').numpy()
    # print(img.shape)

    # ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    plt.imshow(img,vmax=10,vmin=0,cmap="jet")
    plt.savefig(path)

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def random_bernoulli(p:float,shape:int)-> torch.Tensor:
    x = torch.ones((shape,))
    y = torch.bernoulli(torch.tensor(p).expand((shape,)))
    x[y==0] = 0
    return x.item()





def savejet(img,path):
    img = torch.clip(img , -10, 10)
    img = einops.rearrange(img, "b c f h w ->  (b w) ( c f h)")
    lable = img.to('cpu').numpy()
    numpy.save("label.npy",lable)
    plt.imshow(lable, vmax=10, vmin=0, cmap="jet")
    plt.savefig(path, dpi=600, bbox_inches=0, pad_inches=0)





    
    
    

class Get_tager_sample(Dataset):
    def __init__(self, path):
        self.path = os.listdir(path)

    def __getitem__(self, idx):
        img_name = self.path[idx]
        img_f = os.path.join("/media/ps/data/all_png",img_name)
        f = []

        for i in range(16):
            x = cv2.imread(os.path.join(img_f,str(i)+".jpg"))
            f.append(x)
        f = einops.rearrange(numpy.stack(f), "t w h c -> c t w h")



        return f[:,:8,...],f[:,8:16,...]

path = "/media/ps/code/validation"
s = tqdm(os.listdir(path))
for step,i in enumerate (s):

    x = numpy.load(os.path.join(path, i)).astype(numpy.uint8)
    f = []
#
    for s in range(24):
        z = x[s,0,...]
        z = (numpy.clip(z,-1,10)*22.5).astype(numpy.uint8)
        z = cv2.cvtColor(z,cv2.COLOR_GRAY2RGB)
       
        z = Image.fromarray(cv2.cvtColor(z,cv2.COLOR_BGR2RGB))
        f.append(z)
    f = numpy.stack(f)
    imageio.mimwrite("/media/ps/code/validation_grey/"+i[:-3]+"gif",f)