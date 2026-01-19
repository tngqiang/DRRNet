
import torch
from torch.distributions.constraints import lower_cholesky
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import numbers

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'       
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model import  Restormer_CNN_block, GlobalFeatureExtraction, LocalFeatureExtraction

channel =[32,64,16,128]

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=1, embed_dim=16, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Sequential(
            # nn.ReflectionPad2d(1),
            nn.Conv2d(in_c, embed_dim, kernel_size=3,padding=1, bias=bias),
            nn.BatchNorm2d(embed_dim),
            nn.Conv2d(embed_dim, channel[0], kernel_size=3,
                              stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            )
    def forward(self, x):
        x = self.proj(x)
        return x

class Cov1(nn.Module):
    def __init__(self, channel):
        super(Cov1, self).__init__()
        self.cov1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, 3, padding=0),
            )
        self.global_feature = GlobalFeatureExtraction(dim=channel, num_heads = 8)
        self.local_feature = LocalFeatureExtraction(dim=channel)
        self.ffn = nn.Conv2d(channel, channel,kernel_size=3,stride=1, padding=1, bias=False,padding_mode="reflect")
        self.Leaky_relu = nn.LeakyReLU()
    def forward(self, x):
        x1 = self.cov1(x)
        global_feature = self.global_feature(x1)
        local_feature = self.local_feature(x1)
        # x = torch.cat((global_feature, local_feature), dim=1)
        # x = self.Leaky_relu(self.ffn(global_feature+local_feature))
        x = self.Leaky_relu(self.ffn(local_feature))
        return x
  
class Cov4(nn.Module):
    def __init__(self):
        super(Cov4, self).__init__()
        self.cov4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel[0], channel[0], 3, padding=0)
            ) 
        self.global_feature = GlobalFeatureExtraction(dim=channel[0], num_heads = 8)
        self.local_feature = LocalFeatureExtraction(dim=channel[0])
        self.ffn = nn.Conv2d(channel[0], channel[0],kernel_size=3,stride=1, padding=1, bias=False,padding_mode="reflect")
        self.cnn_transfirmer = Restormer_CNN_block(channel[0], channel[0])

    def forward(self, x):
        x1 = self.cov4(x)
        global_feature = self.global_feature(x1)
        local_feature = self.local_feature(x1)
        out = self.ffn(local_feature)

        return out


class Out(nn.Module):
    def __init__(self, dim, out_channels=1, bias=False):
        super(Out, self).__init__()
        self.output = nn.Sequential(
            nn.Conv2d(channel[1]*2, channel[1] , kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(channel[1], out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Tanh()
    def forward(self, x):
        out_enc_level1 = self.output(x)
        return self.sigmoid(out_enc_level1)
        

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pro = OverlapPatchEmbed()
        self.cov1=Cov1(channel[0])
        self.cov2=Cov1(channel[0])
        self.cov3=Cov4()
        self.cov4=Cov4()
        self.sigmoid = nn.Sigmoid()
      
    def forward(self, data_train):
        pro = self.pro(data_train)
        feature_1=self.cov1(pro)
        feature_2=self.cov2(feature_1)
        feature_3=self.cov3(feature_2)
        feature_4=self.cov4(feature_3)
        return feature_1, feature_2,self.sigmoid(feature_3), self.sigmoid(feature_4)

import torch.nn.functional as F
from HLFD import Dense, UNet
from Unet import UNet5, UNet_L

class Restormer_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 2],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[Cov1(channel[1]) for i in range(num_blocks[1])])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Sigmoid()              
    def forward(self, inp_img, base_feature, detail_feature):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0
       
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.avgmax = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.dense=Dense(channel[1])
        # self.unetlow=UNet(channel[1], wave="haar")
        # self.unetlow=UNet_L(channel[1])
        self.unethigh=UNet5(channel[1])
        self.unetlow=UNet_L(channel[1])
        # self.unetlow=UNet5(channel[1])
        self.restormer_decoder = Restormer_Decoder()

        self.out = Out(channel[1]) 
        self.conv = nn.Conv2d(channel[1], 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,pro1, pro2, vi_2, ir_2, vi_3, vi_4, ir_3, ir_4, data_VIS):  #vi_3, vi_4, ir_3, ir_4
        low = torch.cat([vi_4,ir_4], dim=1) + self.avgpool( torch.cat([vi_3,ir_3], dim=1))  
        high = torch.cat([vi_4,ir_4], dim=1) + self.avgmax( torch.cat([vi_3,ir_3], dim=1)) 

        low = self.unetlow(low)
        high = self.unethigh(high)
        img = self.conv(torch.cat([vi_4,ir_4], dim=1))
        out, _ = self.restormer_decoder(data_VIS, low, high)

        return self.conv(low), self.conv(high), out 


