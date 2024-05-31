from modules import UNet_conditional
from train_sample import Diffusion
import torch
import numpy
import einops
import matplotlib.pyplot as plt
device = "cuda"


net = UNet_conditional(c_in=4, c_out=4, device=device)
net.to(device)
diffusion = Diffusion(img_size=256, device=device)
net.load_state_dict(torch.load("checkpoint.pth"))
radar = numpy.load("./dataexample/ex2.npy")
tagert = (radar[0:4, :, :, :] - 0.4202) / 0.8913
lable = (radar[4:8, :, :, :] - 0.4202) / 0.8913

tagert = einops.rearrange(tagert, " t c w h ->  c t w h")
lable = einops.rearrange(lable, " t c w h ->  c t w h")
tagert = torch.from_numpy(tagert)
tagert = einops.repeat(tagert,"c t w h-> a c t w h",a =2)
lable = einops.repeat(lable,"c t w h-> (a c) t w h",a =2)
print(tagert.shape)# b c t c h
tagert = torch.squeeze(tagert).to(device)
# print(x.shape)
sampled_images = diffusion.sample_condition_decoder_(net, n=tagert.shape [0],     image=tagert, )
sampled_images = sampled_images*0.8913+0.4202
lable = lable*0.8913+0.4202
sampled_images = einops.rearrange(sampled_images,"b c w h -> (b w) (c h)")
lable  = einops.rearrange(lable,"b c w h -> (b w) (c h)")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.imshow(sampled_images,cmap="jet",vmin = 0.0,vmax=10)
ax1.set_title("Sampled Images")
ax1.axis("off")

ax2.imshow(lable,cmap="jet",vmin = 0.0,vmax=10)
ax2.set_title("Label")
ax2.axis("off")
plt.tight_layout()
plt.savefig("sampled_images_and_label.png")
# net.load_state_dict(torch.load("atten_128/210.pth"))
