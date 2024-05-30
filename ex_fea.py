import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset
import os
import einops
from fightingcv_attention.attention.TripletAttention import TripletAttention
from einops import rearrange
import numpy
from matplotlib import pyplot as plt
class Get_tager_sample(Dataset):
    def __init__(self, path):
        self.img_path = os.listdir(path)
        self.path = path

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        radar = numpy.load(os.path.join(self.path, img_name))

        # Mytrainsform(radar)
        # radar = trains(torch.from_numpy(radar))

        tagert = (radar[0:4, :, :, :] - 0.4202) / 0.8913
        sample = (radar[4:8, :, :, :] - 0.4202) / 0.8913

        tagert = einops.rearrange(tagert, " t c w h ->  c t w h")
        sample = einops.rearrange(sample, " t c w h ->  c t w h")

        return tagert, sample

    def __len__(self):
        return len(self.img_path)
class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = (nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False))
        #self.conv2 = (nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False))
        self.bn = nn.BatchNorm2d(out_ch)
        # self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.linatt = TripletAttention()
        # self.res_conv = nn.Conv2d(in_ch,out_ch,1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1]<=128:
            x = self.relu(self.bn(self.conv(x)))
            return self.linatt(x)
        return  self.relu(self.bn(self.conv(x)))
        # return  self.bn2(self.relu(self.conv2(self.relu(self.bn(self.conv(x))))))+ self.res_conv(x)


class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.LeakyReLU(self.conv(x))


class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return self.LeakyReLU(self.conv(torch.cat([x1, x2], dim=1)))


class RSU(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()

        assert height >= 2
        self.conv_in = ConvBNReLU(in_ch, out_ch)

        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))

        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))
        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()

            x = m(x, x2)

        return x + x_in


class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=8)])

        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)



        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        self.en_1 = RSU(7,4,16,32)
        # self.g = My_GRU(input_channels=128, output_channels=64)
        self.d1 = DownConvBNReLU(in_ch=32,out_ch=32)
        self.en_2 = RSU(6, 32, 16, 64)

        self.d2 = DownConvBNReLU(in_ch=64, out_ch=64)

        self.en_3 = RSU(5, 64, 16, 128)
        self.d3 = DownConvBNReLU(in_ch=128, out_ch=128)

        self.en_4 = RSU(4, 128, 16, 256)
        self.d4 = DownConvBNReLU(in_ch=256, out_ch=256)


        self.en_5 = RSU4F(256, 16, 512)
        self.d5 = DownConvBNReLU(in_ch=512, out_ch=512)

        self.en_6 = RSU4F(512, 16, 512)
        self.d6 = DownConvBNReLU(in_ch=512, out_ch=512)



    def forward(self,x):
        end = []
        x1 = self.en_1(x)
        end.append(x1)
        x1 = self.d1(x1)

        x2 = self.en_2(x1)
        end.append(x2)
        x2 = self.d2(x2)

        x3 = self.en_3(x2)
        end.append(x3)
        x3 = self.d3(x3)

        x4 = self.en_4(x3)
        end.append(x4)
        x4 = self.d4(x4)

        x5 = self.en_5(x4)
        end.append(x5)
        x5 = self.d5(x5)

        x6 = self.en_6(x5)
        end.append(x6)
        return end




class encoder_start_four(nn.Module):
    def __init__(self):
        super(encoder_start_four, self).__init__()

        self.en_1 = RSU(7,4,16,32)
        # self.g = My_GRU(input_channels=128, output_channels=64)
        self.d1 = DownConvBNReLU(in_ch=32,out_ch=32)
        self.en_2 = RSU(6, 32, 16, 64)

        self.d2 = DownConvBNReLU(in_ch=64, out_ch=64)

        self.en_3 = RSU(5, 64, 16, 128)
        self.d3 = DownConvBNReLU(in_ch=128, out_ch=128)

        self.en_4 = RSU(4, 128, 16, 256)
        self.d4 = DownConvBNReLU(in_ch=256, out_ch=256)


        self.en_5 = RSU4F(256, 16, 512)
        self.d5 = DownConvBNReLU(in_ch=512, out_ch=512)

        self.en_6 = RSU4F(512, 16, 512)
        self.d6 = DownConvBNReLU(in_ch=512, out_ch=512)



    def forward(self,x):
        end = []
        x1 = self.en_1(x)
        end.append(x1)
        x1 = self.d1(x1)

        x2 = self.en_2(x1)
        end.append(x2)
        x2 = self.d2(x2)

        x3 = self.en_3(x2)
        end.append(x3)
        x3 = self.d3(x3)

        x4 = self.en_4(x3)
        end.append(x4)
        x4 = self.d4(x4)

        x5 = self.en_5(x4)
        end.append(x5)
        x5 = self.d5(x5)

        x6 = self.en_6(x5)
        end.append(x6)
        return end



#
# x = torch.ones(size=(4,4,256,256))
# #
# net = encoder()
# #
# x = net(x)
# for i in x:
#     print(i.shape)

# def savejet(img,path,method = "jet"):
#     # img = torch.clip(img , -10, 10)
#     img = einops.rearrange(img, " c  h w ->    h  (c  w)")
#     lable = img.to('cpu').detach().numpy()
#     numpy.save("label.npy",lable)
#     plt.axis("off")
#     plt.imshow(lable, vmax=10, vmin=0, cmap=method)
#     plt.savefig(path, dpi=600, bbox_inches=0, pad_inches=0)



# device  = "cuda:0"

# train_1 = Get_tager_sample("/media/ps/code/train_3w")

# train_1 =  torch.utils.data.DataLoader(train_1,batch_size=4)
# encoder = encoder()
# print(encoder)
# weight =torch.load("evenpth/720.pth")
# # weight = torch.load("model_decoder/2970.pth")
# encoder_weights = encoder.state_dict()
# # # print(encoder)
# filtered_weights = {k: v for k, v in weight.items() if k in encoder_weights}
# encoder.load_state_dict(filtered_weights, strict=False)
# encoder.to(device)
# x,y = next(iter(train_1))
# x = torch.squeeze(x).to(device)
# print(x.shape)
# # savejet(x[1,...],"sdawd0.jpg")

# # x = torch.rand(size=(4,4,256,256)).to("cuda:0")
# out = encoder(x)
# import matplotlib.pyplot as plt
# import numpy as np

# # 创建一个64*128*128的随机数据作为示例





# for stage in range(6):
#     data = out[stage]
#     data = data[1].cpu().detach().numpy()
# # 设置子图的行数和列数
#     num_rows = 8
#     num_cols = (stage+1)*4

#     # 计算每个子图之间的间距
#     hspace = 0.2
#     wspace = 0.2

#     # 创建一个新的图形
#     fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 8),   constrained_layout=True)
#     fig.subplots_adjust(hspace=hspace, wspace=wspace)

#     # 在每个子图中展示一个通道的热力图
#     for i in range(num_rows):
#         for j in range(num_cols):
#             channel = i * num_cols + j
#             if channel < 32*(stage+1):
#                 axes[i, j].imshow(data[channel], cmap='jet')
#                 axes[i, j].set_title(f'Channel {channel}',fontsize=7)
#                 axes[i, j].axis('off')
#             else:
#                 axes[i, j].axis('off')

#     # 显示图形

#     plt.show()

#     plt.savefig('atten/heatmap_no_atten{}.png'.format(stage), dpi=300, bbox_inches='tight')



