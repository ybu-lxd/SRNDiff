import os

import numpy
import torch
import torch.nn as nn
from torch import autocast

from tqdm import tqdm
from matplotlib import pyplot as plt
from multi_train_utils.distributed_utils import is_main_process
from utils import *
import einops
import logging

from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        # print(x.shape)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        # sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        # print(sqrt_alpha_hat.shape)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        # sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]
        # print(sqrt_one_minus_alpha_hat.shape)
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n,in_ch=4):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, in_ch, self.img_size, self.img_size)).to(self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)


                predicted_noise = model(torch.unsqueeze(x,dim=1), t)
                predicted_noise = torch.squeeze(predicted_noise)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        return x
    def sample_condition(self, model, n,x_pre,images,in_ch=1):
        logging.info(f"Sampling {n} new images....")
        model.eval()

        with torch.no_grad():

            x = torch.randn((n, in_ch, self.img_size, self.img_size)).to(self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)

                # predicted_noise = model(torch.unsqueeze(x,dim=1), t,images,x_pre)
                predicted_noise = model(x,t,images)
                # predicted_noise = torch.squeeze(predicted_noise)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        return x

    def sample_condition_decoder(self, model, n,image,label,sq_t,in_ch=4):
        logging.info(f"Sampling {n} new images....")
        model.eval()

        with torch.no_grad():

            x = torch.randn((n, in_ch, self.img_size, self.img_size)).to(self.device)
            # print(images.shape)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                # print(x.shape)
                # predicted_noise = model(torch.unsqueeze(x,dim=1), t,images)
                # print(image.shape,x.shape)
                predicted_noise = model(x, t, image.to(self.device),label.to(self.device),sq_t.to(self.device))
                # predicted_noise = torch.squeeze(predicted_noise)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        x = x*0.8913+0.4202
        return x


    def sample_condition_decoder_(self, model, n,image,label,in_ch=4):
        logging.info(f"Sampling {n} new images....")
        model.eval()

        with torch.no_grad():

            x = torch.randn((n, in_ch, self.img_size, self.img_size)).to(self.device)
            # print(images.shape)
            
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                # print(x.shape)
                # predicted_noise = model(torch.unsqueeze(x,dim=1), t,images)
                # print(image.shape,x.shape)
                predicted_noise = model(x, t, image.to(self.device))
                # predicted_noise = torch.squeeze(predicted_noise)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        # x = x*0.8913+0.4202
        return x


def train(device,model,dataloader,optimizer,diffusion,epoch,scaler,image_size=256):

    mse = nn.MSELoss()
    loss_mean = 0.0

    if is_main_process():
        dataloader = tqdm(dataloader,colour="green",ncols=80)
    for i, (images, lable) in enumerate(dataloader):

        images = images.to(device)
        images = torch.squeeze(images)

        lable = lable.to(device)
        lable = torch.squeeze(lable)
        t = diffusion.sample_timesteps(lable.shape[0]).to(device)
        # with autocast(device_type="cuda",dtype=torch.float16):

        x_t, noise = diffusion.noise_images(lable, t)
        predicted_noise = model(x_t, t,images)
        loss = mse(noise, predicted_noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        loss_mean = loss_mean+loss.item()
        if is_main_process():
            dataloader.desc="epoch: {} loss_mean: {}".format(epoch,round(loss_mean/(i+1),3))

    return loss_mean

def save_images(images, path,lable,flage):

    # grid = torchvision.utils.make_grid(images, **kwargs)
    # print(grid.shape)
    # images = torch.cat([images[0,...],images[0,...],images[0,...],images[0,...]],dim=-1)
    # images = torch.cat([images[1, ...], images[1, ...], images[0, ...], images[0, ...]], dim=-1)
    # images = torch.cat([images[0, ...], images[0, ...], images[0, ...], images[0, ...]], dim=-1)
    # images = torch.cat([images[0, ...], images[0, ...], images[0, ...], images[0, ...]], dim=-1)

    imggrid = []
    lable  = torch.squeeze(lable)
    lable = lable.to("cpu")
    images = images.to("cpu")
    labelgrid = []
    for i in range(images.shape[0]):
        # print(torch.cat([images[i,0,...],images[i,1,...],images[i,2,...],images[i,3,...]],dim=1).shape)
        img  = einops.rearrange(images[i,...],"c w h -> w (c h) ")
        imggrid.append(img)
        label_z = einops.rearrange(lable[i,...],"c w h -> w (c h) ")
        labelgrid.append(label_z)

        # imggrid.append(
        #     # torch.cat([images[i, 0, ...], images[i, 1, ...]], dim=1))
        #     torch.cat([images[i, 0, ...], images[i, 1, ...],images[i, 2, ...],images[i, 3, ...]], dim=1))
        #
        # labelgrid.append( torch.cat([ lable[i, 0, ...], lable[i, 1, ...],lable[i, 2, ...], lable[i, 3, ...] ], dim=1))
        # #labelgrid.append(torch.cat([lable[i, 0, ...], lable[i, 1, ...]], dim=1))








    img = torch.stack(imggrid, dim=0)
    img = einops.rearrange(img, 'c w h -> (c w) h')
    img = img.to('cpu').numpy()



    #
    lable = torch.stack(labelgrid, dim=0)
    lable = einops.rearrange(lable, 'c w h -> (c w) h')
    lable = lable.to('cpu').numpy()
    plt.imshow(lable, vmax=10, vmin=0, cmap="jet")
    plt.savefig("lable.jpg",dpi=1000,bbox_inches=0,pad_inches=0)
    img = numpy.hstack([lable,img])
    plt.imshow(img, vmax=10, vmin=0, cmap="jet")
    plt.savefig(path, dpi=1000, bbox_inches=0, pad_inches=0)



def train_feature(device,model,dataloader,optimizer,diffusion,epoch,feature_2,image_size=256):
    # setup_logging(args.run_name)
    # device = args.device
    # dataloader =  maskedata()
    # model = u2net_lite(out_ch=1,in_ch=1).to(device)
    # optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    mse = nn.MSELoss()

    # logger = SummaryWriter(os.path.join("runs", args.run_name))
    # l = len(dataloader)
    loss_mean = 0.0

    # logging.info(f"Starting epoch {epoch}:")
    if is_main_process():
        dataloader = tqdm(dataloader,colour="green",ncols=80)
    for i, (images, lable) in enumerate(dataloader):
        images = images.to(device)
        # images = torch.squeeze(images)

        lable = lable.to(device)
        lable = torch.squeeze(lable)

        with torch.no_grad():
            feature_2.eval()
            x_pre,x_pre_1  = feature_2(torch.squeeze(images),torch.squeeze(lable))

        x_pre_1 = einops.rearrange(x_pre_1,"b (w h )-> b 1 w h",w = 32)
        t = diffusion.sample_timesteps(lable.shape[0]).to(device)
        # print(x_pre_1.shape)
        x_t, noise = diffusion.noise_images(x_pre_1, t)
        # print(x_t.shape)
        predicted_noise = model(x_t,t,x_pre)
        # print(noise.shape,predicted_noise.shape)
        loss = mse((predicted_noise), noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_mean = loss_mean+loss.item()
        if is_main_process():
            # dataloader.set_postfix(MSE=loss_mean/(i+1))
            dataloader.desc="epoch: {} loss_mean: {}".format(epoch,round(loss_mean/(i+1),3))

    return loss_mean


