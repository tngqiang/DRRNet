# import torch.nn.functional as F
# from functools import partial
# import pywt
# import pywt.data
# import math
# import torch
# from torch import nn
# from einops.layers.torch import Rearrange
# import warnings
# warnings.filterwarnings('ignore')
# '''
# 二次创新模块：WFEConv小波特征增强卷积模块   （冲二，三，保四）
#  WTConv（2024 ECCV顶会）, DEConv（2024 TIP顶刊）或是一些注意力：WFEConv小波特征增强卷积模块 

# 摘要书写思路举例：五步法：任务背景+明确问题动机+点明创新点+简单介绍作用+做实验验证
# 近年来，（什么任务） 是一个具有挑战性的病态问题，简短描述这个任务。  ---交代本文任务背景

# 一些现有的基于深度学习的方法致力于通过增加卷积的深度或宽度来提高模型性能，
# 研究者尝试通过增加卷积神经网络（CNNs）的卷积核大小来模拟视觉变压器（ViTs）
# 自注意力块的全局接收域。              ---目前也有学者在研究卷积CNN针对本文任务（卷积模块作用）

# 卷积神经网络（CNN）结构的学习能力仍未得到充分探索。 ---指明现有的卷积模块不足，暗示本文会提出一种比之前好用卷积模块

# 本文提出一种小波特征增强卷积模块(WFEConv).简单介绍一下这个模块的组成或是作用（来促进特征学习，细节保留等等），
# 及性能（比如轻量、高效），以提高解决本文任务问题能力。
#                                         ---本文的创新点，简单描述创新点，交代解决本文任务的能力
# 在那些公共数据集上，通过做些对比实验和消融实验，验证了该模块或是该方法具有可行性，发展潜力。
#                                         ---通过实验，验证自己的模块或是方法是可行的
# '''

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()

#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1

#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
# class CBAM(nn.Module):
#     def __init__(self, in_planes, ratio=16, kernel_size=7):
#         super(CBAM, self).__init__()
#         self.ca = ChannelAttention(in_planes, ratio)
#         self.sa = SpatialAttention(kernel_size)

#     def forward(self, x):
#         out = x * self.ca(x)
#         result = out * self.sa(out)
#         return result


# def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
#     w = pywt.Wavelet(wave)
#     dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
#     dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
#     dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
#                                dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
#                                dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
#                                dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

#     dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

#     rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
#     rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
#     rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
#                                rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
#                                rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
#                                rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
#     rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
#     return dec_filters, rec_filters
# def wavelet_transform(x, filters):
#     b, c, h, w = x.shape
#     pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
#     x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
#     x = x.reshape(b, c, 4, h // 2, w // 2)
#     return x
# def inverse_wavelet_transform(x, filters):
#     b, c, _, h_half, w_half = x.shape
#     pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
#     x = x.reshape(b, c * 4, h_half, w_half)
#     x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
#     return x

# class Conv2d_cd(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
#                  padding=1, dilation=1, groups=1, bias=False, theta=1.0):
#         super(Conv2d_cd, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#         self.theta = theta

#     def get_weight(self):
#         conv_weight = self.conv.weight
#         conv_shape = conv_weight.shape
#         conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
#         conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
#         conv_weight_cd[:, :, :] = conv_weight[:, :, :]
#         conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
#         conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
#             conv_weight_cd)
#         return conv_weight_cd, self.conv.bias


# class Conv2d_ad(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
#                  padding=1, dilation=1, groups=1, bias=False, theta=1.0):
#         super(Conv2d_ad, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#         self.theta = theta

#     def get_weight(self):
#         conv_weight = self.conv.weight
#         conv_shape = conv_weight.shape
#         conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
#         conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
#         conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
#             conv_weight_ad)
#         return conv_weight_ad, self.conv.bias


# class Conv2d_rd(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
#                  padding=2, dilation=1, groups=1, bias=False, theta=1.0):

#         super(Conv2d_rd, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#         self.theta = theta

#     def forward(self, x):

#         if math.fabs(self.theta - 0.0) < 1e-8:
#             out_normal = self.conv(x)
#             return out_normal
#         else:
#             conv_weight = self.conv.weight
#             conv_shape = conv_weight.shape
#             if conv_weight.is_cuda:
#                 conv_weight_rd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 5 * 5).fill_(0)
#             else:
#                 conv_weight_rd = torch.zeros(conv_shape[0], conv_shape[1], 5 * 5)
#             conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
#             conv_weight_rd[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = conv_weight[:, :, 1:]
#             conv_weight_rd[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -conv_weight[:, :, 1:] * self.theta
#             conv_weight_rd[:, :, 12] = conv_weight[:, :, 0] * (1 - self.theta)
#             conv_weight_rd = conv_weight_rd.view(conv_shape[0], conv_shape[1], 5, 5)
#             out_diff = nn.functional.conv2d(input=x, weight=conv_weight_rd, bias=self.conv.bias,
#                                             stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)

#             return out_diff


# class Conv2d_hd(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
#                  padding=1, dilation=1, groups=1, bias=False, theta=1.0):
#         super(Conv2d_hd, self).__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)

#     def get_weight(self):
#         conv_weight = self.conv.weight
#         conv_shape = conv_weight.shape
#         conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
#         conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
#         conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
#         conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
#             conv_weight_hd)
#         return conv_weight_hd, self.conv.bias


# class Conv2d_vd(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
#                  padding=1, dilation=1, groups=1, bias=False):
#         super(Conv2d_vd, self).__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#     def get_weight(self):
#         conv_weight = self.conv.weight
#         conv_shape = conv_weight.shape
#         conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
#         conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
#         conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
#         conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
#             conv_weight_vd)
#         return conv_weight_vd, self.conv.bias


# class DEConv(nn.Module):
#     def __init__(self, dim):
#         super(DEConv, self).__init__()

#         self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
#         self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
#         self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
#         self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
#         self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

#     def forward(self, x):

#         w1, b1 = self.conv1_1.get_weight()
#         w2, b2 = self.conv1_2.get_weight()
#         w3, b3 = self.conv1_3.get_weight()
#         w4, b4 = self.conv1_4.get_weight()
#         w5, b5 = self.conv1_5.weight, self.conv1_5.bias

#         w = w1 + w2 + w3 + w4 + w5
#         b = b1 + b2 + b3 + b4 + b5
#         res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=1, groups=1)
#         return res

# class WFEConv(nn.Module): #小波特征增强卷积模块
#     def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
#         super(WFEConv, self).__init__()
#         assert in_channels == out_channels
#         self.in_channels = in_channels
#         self.wt_levels = wt_levels
#         self.stride = stride
#         self.dilation = 1

#         self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
#         self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
#         self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

#         self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
#         self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

#         self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
#                                    groups=in_channels, bias=bias)
#         self.base_scale = _ScaleModule([1, in_channels, 1, 1])
#         self.deconv = DEConv(in_channels)
#         self.att = CBAM(in_channels)
#         self.wavelet_convs = nn.ModuleList(
#             [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
#                        groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
#         )
#         self.wavelet_scale = nn.ModuleList(
#             [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
#         )

#         if self.stride > 1:
#             self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
#             self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
#                                                    groups=in_channels)
#         else:
#             self.do_stride = None

#     def forward(self, x):

#         x_ll_in_levels = []
#         x_h_in_levels = []
#         shapes_in_levels = []
#         curr_x_ll = x

#         for i in range(self.wt_levels):
#             curr_shape = curr_x_ll.shape
#             shapes_in_levels.append(curr_shape)
#             if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
#                 curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
#                 curr_x_ll = F.pad(curr_x_ll, curr_pads)

#             curr_x = self.wt_function(curr_x_ll)
#             curr_x_ll = curr_x[:, :, 0, :, :]

#             shape_x = curr_x.shape
#             curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
#             curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
#             curr_x_tag = curr_x_tag.reshape(shape_x)

#             x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
#             x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])
#         next_x_ll = 0
#         for i in range(self.wt_levels - 1, -1, -1):
#             curr_x_ll = x_ll_in_levels.pop()
#             curr_x_h = x_h_in_levels.pop()
#             curr_shape = shapes_in_levels.pop()

#             curr_x_ll = curr_x_ll + next_x_ll

#             curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
#             next_x_ll = self.iwt_function(curr_x)

#             next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

#         x_tag = next_x_ll
#         assert len(x_ll_in_levels) == 0
#         # x = self.base_scale(self.base_conv(x))
#         x = self.base_scale(self.deconv(x)) #使用DEconv顶会卷积去替换普通卷积，
#         x = x + x_tag
#         x = self.att(x)    #增加一个CBAM注意力模块,增强全局特征和局部特征表达，注意力模块可自行替换其它
#         if self.do_stride is not None:
#             x = self.do_stride(x)
#         return x
# class _ScaleModule(nn.Module):
#     def __init__(self, dims, init_scale=1.0, init_bias=0):
#         super(_ScaleModule, self).__init__()
#         self.dims = dims
#         self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
#         self.bias = None
#     def forward(self, x):
#         return torch.mul(self.weight, x)


# # 输入 N C H W,  输出 N C H W
# # if __name__ == '__main__':
# #     block = WFEConv(32,32).cuda()
# #     input = torch.rand(1, 32, 64, 64).cuda()
# #     output = block(input)
# #     print("input.shape:", input.shape)
# #     print("output.shape:",output.shape)
    
# ###############################################

# import torch
# import torch.nn as nn
# from pytorch_wavelets import DWTForward


# class Down_wt(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(Down_wt, self).__init__()
#         self.wt = DWTForward(J=1, mode='zero', wave='haar')
#         self.conv_bn_relu = nn.Sequential(
#             nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         yL, yH = self.wt(x)
#         y_HL = yH[0][:, :, 0, ::]
#         y_LH = yH[0][:, :, 1, ::]
#         y_HH = yH[0][:, :, 2, ::]
#         x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
#         x = self.conv_bn_relu(x)
#         return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # https://github.com/Zheng-MJ/SMFANet
# '''
# SMFANet：一种轻量级的自调制特征聚合网络，以实现高效的图像超分辨率   ECCV 2024 顶会

# 即插即用特征融合模块：SMFA
# 探索非局部信息，从而更好地进行高分辨率的图像重建。
# 然而，SA机制需要大量的计算资源，这限制了其在低功耗设备中的应用。
# 此外，SA机制限制了其捕获局部细节的能力，从而影响图像重建效果。
# 为了解决这些问题，我们提出了一个自研特征融合（SMFA）模块，
# 以协同利用局部和非局部特征交互来进行更准确的高分辨率的图像重建。

# 具体来说，SMFA模块采用了一种有效的自注意近似（EASA）分支来捕获非局部信息，
# 并使用一个局部细节估计（LDE）分支来捕获局部细节。

# 适用于：高分辨率图像重建，暗光增强，图像恢复，等所有CV任务上通用特征融合模块
# '''
# # class DMlp(nn.Module):
# #     def __init__(self, dim, growth_rate=2.0):
# #         super().__init__()
# #         hidden_dim = int(dim * growth_rate)
# #         self.conv_0 = nn.Sequential(
# #             nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
# #             nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
# #         )
# #         self.act = nn.GELU()
# #         self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

# #     def forward(self, x):
# #         x = self.conv_0(x)
# #         x = self.act(x)
# #         x = self.conv_1(x)
# #         return x

# class Mlp(nn.Module):
#     """
#     MLP as used in Vision Transformer, MLP-Mixer and related networks
#     """
#     def __init__(self, 
#                  in_features, 
#                  hidden_features=None, 
#                  ffn_expansion_factor = 2,
#                  bias = False):
#         super().__init__()
#         hidden_features = int(in_features*ffn_expansion_factor)

#         self.project_in = nn.Conv2d(
#             in_features, hidden_features*2, kernel_size=1, bias=bias)

#         self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
#                                 stride=1, padding=1, groups=hidden_features, bias=bias)

#         self.project_out = nn.Conv2d(
#             hidden_features, in_features, kernel_size=1, bias=bias)
#     def forward(self, x):
#         x = self.project_in(x)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         x = F.gelu(x1) * x2
#         x = self.project_out(x)
#         return x



# class PCFN(nn.Module):
#     def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
#         super().__init__()
#         hidden_dim = int(dim * growth_rate)
#         p_dim = int(hidden_dim * p_rate)
#         self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
#         self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1)

#         self.act = nn.GELU()
#         self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

#         self.p_dim = p_dim
#         self.hidden_dim = hidden_dim

#     def forward(self, x):
#         if self.training:
#             x = self.act(self.conv_0(x))
#             x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)
#             x1 = self.act(self.conv_1(x1))
#             x = self.conv_2(torch.cat([x1, x2], dim=1))
#         else:
#             x = self.act(self.conv_0(x))
#             x[:, :self.p_dim, :, :] = self.act(self.conv_1(x[:, :self.p_dim, :, :]))
#             x = self.conv_2(x)
#         return x
# class SMFA(nn.Module):
#     def __init__(self, dim=36):
#         super(SMFA, self).__init__()
#         self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
#         self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
#         self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

#         self.lde = Mlp(dim, 2)

#         self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

#         self.gelu = nn.GELU()
#         self.down_scale = 8

#         self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
#         self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

#     def forward(self, f):
#         _, _, h, w = f.shape
#         y, x = self.linear_0(f).chunk(2, dim=1)
#         x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
#         x_v = torch.var(x, dim=(-2, -1), keepdim=True)
#         x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w),
#                                 mode='nearest')
#         y_d = self.lde(y)
#         return self.linear_2(x_l + y_d)


# import torch
# import torch.nn as nn
# from einops.layers.torch import Rearrange
# '''
# 来自CVPR 2024 顶会 
# 即插即用模块：CVIM 跨视图交互模块 （特征融合模块）
# 含二次创新模块 EAGFM 有效注意力引导特征融合模块 
# EGAFM是CVIM模块发二次创新，效果优于CVIM ，可以直接拿去发小论文，冲SCI一区或是二区
 
# CVIM模块的主要目标是通过有效的左右视图交互，实现图像的跨视图信息融合，
# 从而提高图像超分辨率的性能，增强图像细节特征。具体来说：
# 1.增强跨视图信息共享：通过左右视图之间的特征交互，提取视图之间的互补信息。
# 2.减少计算复杂度：通过优化输入特征的维度和去除冗余操作，显著降低了计算开销，使其适合于轻量化网络。
# 3.提高跨视图融合效率：结合左右视图的特征，提高重建图像细节和纹理的准确性。
# CVIM通过轻量化的跨视图交互机制，高效特征提取和融合左右视图的互补信息，
# 显著提升图像的超分辨率性能，同时保持低计算复杂度和高效集成能力。

# 特征融合模块适用于所有计算机视觉CV任务，通用的即插即用模块
# '''
# class PixelAttention(nn.Module):
#     def __init__(self, dim):
#         super(PixelAttention, self).__init__()
#         self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, pattn1):
#         B, C, H, W = x.shape
#         x = x.unsqueeze(dim=2)  # B, C, 1, H, W
#         pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
#         x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
#         x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
#         pattn2 = self.pa2(x2)
#         pattn2 = self.sigmoid(pattn2)
#         return pattn2
# class CVIM(nn.Module):

#     def __init__(self, c):
#         super().__init__()
#         self.scale = c ** -0.5

#         self.l_proj1 = nn.Sequential(
#             nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
#         )
#         self.r_proj1 = nn.Sequential(
#             nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
#         )

#         self.l_proj2 = nn.Sequential(
#             nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
#         )
#         self.r_proj2 = nn.Sequential(
#             nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
#         )

#         self.l_proj3 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
#         self.r_proj3 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

#     def forward(self, x_l, x_r):
#         Q_l = self.l_proj1(x_l).permute(0, 2, 3, 1)  # B, H, W, c
#         Q_r_T = self.r_proj1(x_r).permute(0, 2, 1, 3)  # B, H, c, W (transposed)

#         V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
#         V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

#         # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
#         attention = torch.matmul(Q_l, Q_r_T) * self.scale

#         F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
#         F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)  # B, H, W, c

#         # scale
#         F_r2l = self.l_proj3(F_r2l.permute(0, 3, 1, 2))
#         F_l2r = self.r_proj3(F_l2r.permute(0, 3, 1, 2))
#         return x_l + F_r2l+x_r + F_l2r
# # 顶会二次创新模块 EAGFM 有效注意力引导融合模块  可以直接去发小论文 冲sci一区和二区  高效发小论文
# # 需要的小伙伴的可以私信我发给你

# # 输入 N C H W,  输出 N C H W
# # if __name__ == '__main__':
# #     input1 = torch.randn(1, 32, 64, 64)
# #     input2 = torch.randn(1, 32, 64, 64)
# #     # 初始化CVIM模块并设定通道维度
# #     CVIM_module = CVIM(32)
# #     output =CVIM_module(input1,input2)
# #     # 输出结果的形状
# #     print("CVIM_输入张量的形状：", input1.shape)
# #     print("CVIM_输出张量的形状：", output.shape)

# from einops import rearrange
# import torch
# from torch.nn import functional as F
# from torch import nn
# # https://openaccess.thecvf.com/content/CVPR2023/papers/Jin_DNF_Decouple_and_Feedback_Network_for_Seeing_in_the_Dark_CVPR_2023_paper.pdf
# # https://github.com/Srameo/DNF/tree/main
# '''
# CVPR 2023顶会

# 本文介绍了一个新颖的网络架构——Decouple and Feedback Network (DNF)，
# 其目标是解决现有方法在基于RAW图像的低光图像增强任务中的性能瓶颈。
# 为了应对单阶段方法的域歧义问题以及多阶段方法的信息丢失问题，
# DNF网络提出了两个主要的创新点：域特定任务解耦 和去噪反馈机制 。
# DNF网络框架由：CID ，MCC，GFM

# CID 模块的作用： CID 模块负责在 RAW 域内执行独立的去噪任务。由于 RAW 图像中的噪声通常是信号无关的，
# 并且各个通道间的噪声分布独立，因此 CID 模块使用了深度卷积（7x7卷积核）来移除通道独立的噪声。
# 每个 CID 模块独立处理各通道的噪声，并且通过残差结构进一步增强去噪效果。该模块的引入保证了 RAW 图像的去噪精度。

# MCC 模块的作用： MCC 模块专注于 RAW 到 sRGB 的颜色转换任务。在图像信号处理 (ISP) 流水线中，
# 颜色转换通常通过通道级的矩阵变换实现。MCC 模块通过1x1卷积层和3x3深度卷积层生成Q、K和V矩阵，
# 进行颜色空间转换和局部细节的优化，确保最终 sRGB 图像的颜色校正效果。

# GFM 模块的作用： GFM 模块用于实现特征级别的信息反馈，将 RAW 解码器输出的去噪特征重新注入到 RAW 编码器中，
# 以改善去噪效果。通过门控机制，GFM 能够自适应地融合反馈特征和初始去噪特征，进而使得网络在噪声与细节之间进行有效区分，
# 从而减少去噪过程中的细节丢失。

# 总结：
# CID 模块通过独立通道的去噪提升了 RAW 图像中的去噪能力，
# MCC 模块则通过矩阵变换实现了高效的颜色校正，
# GFM 模块通过门控机制进行特征融合，解决了传统多阶段方法中的信息丢失问题。实现了更高效的低光图像增强。

# 即插即用模块适用于：图像增强，低光图像增强，图像去噪，图像恢复，低光目标检测，低光图像分割等所有CV任务通用模块
# '''
# class LayerNorm(nn.Module):
#     r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
#     The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
#     shape (batch_size, height, width, channels) while channels_first corresponds to inputs
#     with shape (batch_size, channels, height, width).
#     """

#     def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.eps = eps
#         self.data_format = data_format
#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise NotImplementedError
#         self.normalized_shape = (normalized_shape, )

#     def forward(self, x):
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         elif self.data_format == "channels_first":
#             u = x.mean(1, keepdim=True)
#             s = (x - u).pow(2).mean(1, keepdim=True)
#             x = (x - u) / torch.sqrt(s + self.eps)
#             x = self.weight[:, None, None] * x + self.bias[:, None, None]
#             return x
# # CI
# class DConv7(nn.Module):
#     def __init__(self, f_number, padding_mode='reflect') -> None:
#         super().__init__()
#         self.dconv = nn.Conv2d(f_number, f_number, kernel_size=7, padding=3, groups=f_number, padding_mode=padding_mode)

#     def forward(self, x):
#         return self.dconv(x)

# # Post-CI
# class MLP(nn.Module):
#     def __init__(self, f_number, excitation_factor=2) -> None:
#         super().__init__()
#         self.act = nn.GELU()
#         self.pwconv1 = nn.Conv2d(f_number, excitation_factor * f_number, kernel_size=1)
#         self.pwconv2 = nn.Conv2d(f_number * excitation_factor, f_number, kernel_size=1)

#     def forward(self, x):
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         return x

# class CID(nn.Module):
#     def __init__(self, f_number, padding_mode) -> None:
#         super().__init__()
#         self.channel_independent = DConv7(f_number, padding_mode)
#         self.channel_dependent = MLP(f_number, excitation_factor=2)

#     def forward(self, x):
#         return self.channel_dependent(self.channel_independent(x))

# class MCC(nn.Module):
#     def __init__(self, f_number, num_heads, padding_mode, bias=False) -> None:
#         super().__init__()
#         self.norm = LayerNorm(f_number, eps=1e-6, data_format='channels_first')

#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#         self.pwconv = nn.Conv2d(f_number, f_number * 3, kernel_size=1, bias=bias)
#         self.dwconv = nn.Conv2d(f_number * 3, f_number * 3, 3, 1, 1, bias=bias, padding_mode=padding_mode, groups=f_number * 3)
#         self.project_out = nn.Conv2d(f_number, f_number, kernel_size=1, bias=bias)
#         self.feedforward = nn.Sequential(
#             nn.Conv2d(f_number, f_number, 1, 1, 0, bias=bias),
#             nn.GELU(),
#             nn.Conv2d(f_number, f_number, 3, 1, 1, bias=bias, groups=f_number, padding_mode=padding_mode),
#             nn.GELU()
#         )

#     def forward(self, x):
#         attn = self.norm(x)
#         _, _, h, w = attn.shape

#         qkv = self.dwconv(self.pwconv(attn))
#         q, k, v = qkv.chunk(3, dim=1)

#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)

#         out = (attn @ v)

#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

#         out = self.project_out(out)
#         out = self.feedforward(out + x)
#         return out

# class GFM(nn.Module):
#     def __init__(self, in_channels, feature_num=2, bias=True, padding_mode='reflect', **kwargs) -> None:
#         super().__init__()
#         self.feature_num = feature_num

#         hidden_features = in_channels * feature_num
#         self.pwconv = nn.Conv2d(hidden_features, hidden_features * 2, 1, 1, 0, bias=bias)
#         self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, 1, 1, bias=bias, padding_mode=padding_mode, groups=hidden_features * 2)
#         self.project_out = nn.Conv2d(hidden_features, in_channels, kernel_size=1, bias=bias)
#         self.mlp = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=True)

#     def forward(self, *inp_feats):
#         assert len(inp_feats) == self.feature_num
#         shortcut = inp_feats[0]
#         x = torch.cat(inp_feats, dim=1)
#         x = self.pwconv(x)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         x = F.gelu(x1) * x2
#         x = self.project_out(x)
#         return self.mlp(x + shortcut)

# # # 输入 N C D H W,  输出 N C D H W
# # if __name__ == '__main__':
# #     # 创建一个简单的输入特征图
# #     input1 = torch.randn(2, 64, 32, 32)
# #     input2 = torch.randn(2, 64, 32, 32)
# #     # 创建一个 GFM 实例
# #     GFM = GFM(in_channels=64)
# #     # 将两个输入特征图传递给 GFM 模块
# #     output = GFM(input1, input2)
# #     # 打印输入和输出的尺寸
# #     print(f"input 1 shape: {input1.shape}")
# #     print(f"input 2 shape: {input2.shape}")
# #     print(f"output shape: {output.shape}")
# # --------------------------------------------------------
# # 论文：DEA-Net: Single image dehazing based on detail enhanced convolution and content-guided attention
# # GitHub地址：https://github.com/cecret3350/DEA-Net/tree/main
# # --------------------------------------------------------
# '''
# DEA-Net：基于细节增强卷积和内容引导注意力的单图像去雾 (IEEE TIP 2024顶会论文)

# 我们提出了一种新的注意机制，称为轮廓引导注意（CGA），以一种从粗到细的方式生成特定于通道的sim。
# 通过使用输入的特征引导SIM的生成，CGA为每个通道分配唯一的SIM，
# 使模型关注每个通道的重要区域。因此，可以强调用特征编码的更多有用的信息，以有效地提高性能。
# 此外，还提出了一种基于cga的混合融合方案，可以有效地将编码器部分的低级特征与相应的高级特征进行融合。
# '''
# import torch
# from torch import nn
# from einops.layers.torch import Rearrange

# class SpatialAttention1(nn.Module):
#     def __init__(self):
#         super(SpatialAttention1, self).__init__()
#         self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

#     def forward(self, x):
#         x_avg = torch.mean(x, dim=1, keepdim=True)
#         x_max, _ = torch.max(x, dim=1, keepdim=True)
#         x2 = torch.cat([x_avg, x_max], dim=1)
#         sattn = self.sa(x2)
#         return sattn

# class ChannelAttention1(nn.Module):
#     def __init__(self, dim, reduction=8):
#         super(ChannelAttention1, self).__init__()
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.ca = nn.Sequential(
#             nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
#         )

#     def forward(self, x):
#         x_gap = self.gap(x)
#         cattn = self.ca(x_gap)
#         return cattn


# class PixelAttention(nn.Module):
#     def __init__(self, dim):
#         super(PixelAttention, self).__init__()
#         self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, pattn1):
#         B, C, H, W = x.shape
#         x = x.unsqueeze(dim=2)  # B, C, 1, H, W
#         pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
#         x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
#         x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
#         pattn2 = self.pa2(x2)
#         pattn2 = self.sigmoid(pattn2)
#         return pattn2

# class CGAFusion(nn.Module):
#     def __init__(self, dim, reduction=8):
#         super(CGAFusion, self).__init__()
#         self.sa = SpatialAttention1()
#         self.ca = ChannelAttention1(dim, reduction)
#         self.pa = PixelAttention(dim)
#         self.conv = nn.Conv2d(dim, dim, 1, bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, y):
#         initial = x + y
#         cattn = self.ca(initial)
#         sattn = self.sa(initial)
#         pattn1 = sattn + cattn
#         pattn2 = self.sigmoid(self.pa(initial, pattn1))
#         result = initial + pattn2 * x + (1 - pattn2) * y
#         result = self.conv(result)
#         return result

# # 双分支特征融合
# # # 输入 N C H W,  输出 N C H W
# # if __name__ == '__main__':
# #     block = CGAFusion(32)
# #     input1 = torch.rand(3, 32, 64, 64)
# #     input2 = torch.rand(3, 32, 64, 64)
# #     output = block(input1, input2)
# #     print(output.size())

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numbers
# from einops import rearrange

# '''
# 来自CVPR顶会论文      
# 两个即插即用模块：MDTA 和 GDFN
#            将MDTA和GDFN两个模块结合：取名为 MGDB

# 由于卷积神经网络（CNNs）能够从大规模数据中学习可泛化的图像先验，这些模型已被广泛应用于图像恢复及相关任务中。
# 最近，另一类神经架构——Transformer，在自然语言处理和高级视觉任务上展现出了显著的性能提升。
# 虽然Transformer模型缓解了CNNs的缺点（例如，有限的感受野和对输入内容的不适应性），
# 但其计算复杂度随空间分辨率的增大而呈二次增长，因此，对于大多数涉及高分辨率图像的图像恢复任务来说，
# Transformer模型的应用是不可行的。

# 在这项工作中，我们提出了一种高效的Transformer模型，通过对构建块（多头注意力和前馈网络）进行几项关键设计，
# 使其能够捕捉长距离的像素交互，同时仍然适用于大图像。我们的模型，在多个图像恢复任务上实现了最先进的性能，
# 包括图像去雨、单图像运动去模糊、散焦去模糊（单图像和双像素数据）以及图像去噪（高斯灰度/彩色去噪和真实图像去噪）。

# MDTA模块的主要作用包括：
# 1.线性复杂度：通过将自注意力机制应用于特征维度而非空间维度，MDTA模块显著降低了计算复杂度，
#          使其具有线性复杂度。这使得MDTA模块能够高效地处理高分辨率图像。
# 2.全局上下文建模：虽然MDTA模块在空间维度上不显式地建模像素对之间的交互，但它通过计算特征通道之间的协方差来生成注意力图，
#          从而隐式地编码全局上下文信息。这使得模型能够在不牺牲全局感受野的情况下，高效地捕捉图像中的长距离依赖关系。
# 3.局部上下文混合：MDTA模块在计算注意力图之前，通过1x1卷积和深度可分离卷积对输入特征进行局部上下文混合。
#          这有助于强调空间局部上下文，并将卷积操作的互补优势融入到模型中。

# GDFN模块的主要作用包括：
# 1.受控特征转换：通过引入门控机制，GDFN模块能够控制哪些互补特征应该向前流动，
#         并允许网络层次结构中的后续层专注于更精细的图像属性。这有助于生成高质量的输出图像。
# 2.局部内容混合：与MDTA模块类似，GDFN模块也包含深度可分离卷积，用于编码来自空间相邻像素位置的信息。
#             这有助于学习局部图像结构，对于有效的图像恢复至关重要。
# 上述模块适用于：图像恢复，图像去模糊，图像去噪，目标检测，图像分割，图像分类等所有计算机视觉CV任务通用的即插即用模块
# '''


# def to_3d(x):
#     return rearrange(x, 'b c h w -> b (h w) c')


# def to_4d(x, h, w):
#     return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

# class BiasFree_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(BiasFree_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)

#         assert len(normalized_shape) == 1

#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.normalized_shape = normalized_shape

#     def forward(self, x):
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return x / torch.sqrt(sigma + 1e-5) * self.weight


# class WithBias_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(WithBias_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)

#         assert len(normalized_shape) == 1

#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.normalized_shape = normalized_shape

#     def forward(self, x):
#         mu = x.mean(-1, keepdim=True)
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


# class LayerNorm(nn.Module):
#     def __init__(self, dim, LayerNorm_type= 'WithBias'):
#         super(LayerNorm, self).__init__()
#         if LayerNorm_type == 'BiasFree':
#             self.body = BiasFree_LayerNorm(dim)
#         else:
#             self.body = WithBias_LayerNorm(dim)

#     def forward(self, x):
#         h, w = x.shape[-2:]
#         return to_4d(self.body(to_3d(x)), h, w)


# ##########################################################################
# ## Gated-Dconv Feed-Forward Network (GDFN)
# class GDFN(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
#         super(GDFN, self).__init__()

#         hidden_features = int(dim * ffn_expansion_factor)

#         self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

#         self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
#                                 groups=hidden_features * 2, bias=bias)

#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

#     def forward(self, x):
#         x = self.project_in(x)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         x = F.gelu(x1) * x2
#         x = self.project_out(x)
#         return x


# ##########################################################################
# ## Multi-DConv Head Transposed Self-Attention (MDTA)
# class MDTA(nn.Module):
#     def __init__(self, dim, num_heads=4, bias=False):
#         super(MDTA, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
#         self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

#     def forward(self, x):
#         b, c, h, w = x.size()

#         qkv = self.qkv_dwconv(self.qkv(x))
#         q, k, v = qkv.chunk(3, dim=1)

#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)

#         out = (attn @ v)

#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

#         out = self.project_out(out)
#         return out
# class MGDB(nn.Module): #TransformerBlock
#     def __init__(self, dim):
#         super(MGDB, self).__init__()
#         self.norm1 = LayerNorm(dim)
#         self.attn = MDTA(dim)
#         self.norm2 = LayerNorm(dim)
#         self.ffn = GDFN(dim)

#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.ffn(self.norm2(x))
#         return x

# # 输入 B C H W,  输出 B C H W
# # if __name__ == '__main__':
# #     # # 生成随机输入张量
# #     input = torch.randn(1, 32, 64, 64)
# #     # 实例化模型对象
# #     MDTA_model = MDTA(dim=32)
# #     GDFN_model = GDFN(dim=32)
# #     MGDB_model = MGDB(dim=32)  # MGDB 是MDTA和GDFN模块的结合

# #     # 执行 MDTA 前向传播
# #     output = MDTA_model(input)
# #     print('MDTA_input_size:', input.size())
# #     print('MDTA_output_size:', output.size())

# #     # 执行 GDFN 前向传播
# #     output = GDFN_model(input)
# #     print('GDFN_input_size:', input.size())
# #     print('GDFN_output_size:', output.size())

# #     # 执行 MGDB 前向传播
# #     output = MGDB_model(input)
# #     print('MGDB_input_size:', input.size())
# #     print('MGDB_output_size:', output.size())

# # TPAMI 2024：Frequency-aware Feature Fusion for Dense Image Prediction
# #https://github.com/Linwei-Chen/FreqFusion/blob/main/FreqFusion.py
# # https://arxiv.org/pdf/2408.12879

# '''
# 即插即用模块：FreqFusion 多尺度特征融合    TPAMI 2024顶刊

# 密集图像预测任务需要具有强大类别信息和高分辨率精确空间边界细节的特征。为了实现这一点，
# 现代分层模型通常利用特征融合，直接相加来自深层的上采样粗粒度特征和来自较低级别的高分辨率特征。
# 在本文中，我们观察到对象内融合特征值的快速变化，由于受到干扰的高频特征导致类别内不一致。
# 此外，融合特征中的模糊边界缺乏准确的高频，从而导致边界位移。

# 传统的特征融合方法在语义分割、目标检测、实例分割等任务中
# 通过简单地上采样粗粒度特征并将其与高分辨率的低级特征直接相加，
# 但这样做会导致类别内特征不一致和模糊的边界。

# FreqFusion模块主要包含三个核心生成器：

# 自适应低通滤波生成器（ALPF ）：通过预测空间变化的低通滤波器，平滑上采样高层特征，
# 减少高频分量在对象内的扰动，以减少类别内不一致性。

# 偏移生成器（Offset）：通过计算局部相似性，预测偏移量并重新采样特征，
# 以使用类别内相似性更高的邻近特征替换不一致的特征，从而修正不一致的区域，尤其是大范围不一致的区域以及细小的边界区域。

# 自适应高通滤波生成器（AHPF）：通过空间变化的高通滤波器增强低级特征的高频信息，
# 特别是恢复在下采样过程中丢失的边界细节，以提高边界清晰度。

# FreqFusion模块设计的目的是通过这些生成器在特征融合过程中同时保持类别内特征一致性和边界清晰度，
# 从而提升密集图像预测任务中的性能。实验结果表明，该模块在多个任务中都取得了显著的性能提升。

# 这个特征融合模块：适用于所以CV任务
# '''
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.ops.carafe import normal_init, xavier_init, carafe
# from torch.utils.checkpoint import checkpoint
# import warnings
# import numpy as np
# import warnings
# warnings.filterwarnings('ignore')

# def normal_init(module, mean=0, std=1, bias=0):
#     if hasattr(module, 'weight') and module.weight is not None:
#         nn.init.normal_(module.weight, mean, std)
#     if hasattr(module, 'bias') and module.bias is not None:
#         nn.init.constant_(module.bias, bias)


# def constant_init(module, val, bias=0):
#     if hasattr(module, 'weight') and module.weight is not None:
#         nn.init.constant_(module.weight, val)
#     if hasattr(module, 'bias') and module.bias is not None:
#         nn.init.constant_(module.bias, bias)


# def resize(input,
#            size=None,
#            scale_factor=None,
#            mode='nearest',
#            align_corners=None,
#            warning=True):
#     if warning:
#         if size is not None and align_corners:
#             input_h, input_w = tuple(int(x) for x in input.shape[2:])
#             output_h, output_w = tuple(int(x) for x in size)
#             if output_h > input_h or output_w > input_w:
#                 if ((output_h > 1 and output_w > 1 and input_h > 1
#                      and input_w > 1) and (output_h - 1) % (input_h - 1)
#                         and (output_w - 1) % (input_w - 1)):
#                     warnings.warn(
#                         f'When align_corners={align_corners}, '
#                         'the output would more aligned if '
#                         f'input size {(input_h, input_w)} is `x+1` and '
#                         f'out size {(output_h, output_w)} is `nx+1`')
#     return F.interpolate(input, size, scale_factor, mode, align_corners)


# def hamming2D(M, N):
#     """
#     生成二维Hamming窗

#     参数：
#     - M：窗口的行数
#     - N：窗口的列数

#     返回：
#     - 二维Hamming窗
#     """
#     # 生成水平和垂直方向上的Hamming窗
#     # hamming_x = np.blackman(M)
#     # hamming_x = np.kaiser(M)
#     hamming_x = np.hamming(M)
#     hamming_y = np.hamming(N)
#     # 通过外积生成二维Hamming窗
#     hamming_2d = np.outer(hamming_x, hamming_y)
#     return hamming_2d


# class FreqFusion(nn.Module):
#     def __init__(self,
#                  hr_channels,
#                  lr_channels,
#                  scale_factor=1,
#                  lowpass_kernel=5,
#                  highpass_kernel=3,
#                  up_group=1,
#                  encoder_kernel=3,
#                  encoder_dilation=1,
#                  compressed_channels=64,
#                  align_corners=False,
#                  upsample_mode='nearest',
#                  feature_resample=True,  # use offset generator or not
#                  feature_resample_group=4,
#                  comp_feat_upsample=True,  # use ALPF & AHPF for init upsampling
#                  use_high_pass=True,
#                  use_low_pass=True,
#                  hr_residual=True,
#                  semi_conv=True,
#                  hamming_window=True,  # for regularization, do not matter really
#                  feature_resample_norm=True,
#                  **kwargs):
#         super().__init__()
#         self.scale_factor = scale_factor
#         self.lowpass_kernel = lowpass_kernel
#         self.highpass_kernel = highpass_kernel
#         self.up_group = up_group
#         self.encoder_kernel = encoder_kernel
#         self.encoder_dilation = encoder_dilation
#         self.compressed_channels = compressed_channels
#         self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
#         self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)
#         self.content_encoder = nn.Conv2d(  # ALPF generator
#             self.compressed_channels,
#             lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
#             self.encoder_kernel,
#             padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
#             dilation=self.encoder_dilation,
#             groups=1)

#         self.align_corners = align_corners
#         self.upsample_mode = upsample_mode
#         self.hr_residual = hr_residual
#         self.use_high_pass = use_high_pass
#         self.use_low_pass = use_low_pass
#         self.semi_conv = semi_conv
#         self.feature_resample = feature_resample
#         self.comp_feat_upsample = comp_feat_upsample
#         if self.feature_resample:
#             self.dysampler = LocalSimGuidedSampler(in_channels=compressed_channels, scale=2, style='lp',
#                                                    groups=feature_resample_group, use_direct_scale=True,
#                                                    kernel_size=encoder_kernel, norm=feature_resample_norm)
#         if self.use_high_pass:
#             self.content_encoder2 = nn.Conv2d(  # AHPF generator
#                 self.compressed_channels,
#                 highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
#                 self.encoder_kernel,
#                 padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
#                 dilation=self.encoder_dilation,
#                 groups=1)
#         self.hamming_window = hamming_window
#         lowpass_pad = 0
#         highpass_pad = 0
#         if self.hamming_window:
#             self.register_buffer('hamming_lowpass', torch.FloatTensor(
#                 hamming2D(lowpass_kernel + 2 * lowpass_pad, lowpass_kernel + 2 * lowpass_pad))[None, None,])
#             self.register_buffer('hamming_highpass', torch.FloatTensor(
#                 hamming2D(highpass_kernel + 2 * highpass_pad, highpass_kernel + 2 * highpass_pad))[None, None,])
#         else:
#             self.register_buffer('hamming_lowpass', torch.FloatTensor([1.0]))
#             self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))
#         self.init_weights()

#     def init_weights(self):
#         for m in self.modules():
#             # print(m)
#             if isinstance(m, nn.Conv2d):
#                 xavier_init(m, distribution='uniform')
#         normal_init(self.content_encoder, std=0.001)
#         if self.use_high_pass:
#             normal_init(self.content_encoder2, std=0.001)

#     def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
#         if scale_factor is not None:
#             mask = F.pixel_shuffle(mask, self.scale_factor)
#         n, mask_c, h, w = mask.size()
#         mask_channel = int(mask_c / float(kernel ** 2))
#         # mask = mask.view(n, mask_channel, -1, h, w)
#         # mask = F.softmax(mask, dim=2, dtype=mask.dtype)
#         # mask = mask.view(n, mask_c, h, w).contiguous()

#         mask = mask.view(n, mask_channel, -1, h, w)
#         mask = F.softmax(mask, dim=2, dtype=mask.dtype)
#         mask = mask.view(n, mask_channel, kernel, kernel, h, w)
#         mask = mask.permute(0, 1, 4, 5, 2, 3).reshape(n, -1, kernel, kernel)
#         # mask = F.pad(mask, pad=[padding] * 4, mode=self.padding_mode) # kernel + 2 * padding
#         mask = mask * hamming
#         mask /= mask.sum(dim=(-1, -2), keepdims=True)
#         # print(hamming)
#         # print(mask.shape)
#         mask = mask.view(n, mask_channel, h, w, -1)
#         mask = mask.permute(0, 1, 4, 2, 3).reshape(n, -1, h, w).contiguous()
#         return mask

#     def forward(self, hr_feat, lr_feat, use_checkpoint=False):
#         if use_checkpoint:
#             return checkpoint(self._forward, hr_feat, lr_feat)
#         else:
#             return self._forward(hr_feat, lr_feat)

#     def _forward(self, hr_feat, lr_feat):
#         compressed_hr_feat = self.hr_channel_compressor(hr_feat)
#         compressed_lr_feat = self.lr_channel_compressor(lr_feat)
#         if self.semi_conv:
#             if self.comp_feat_upsample:
#                 if self.use_high_pass:
#                     mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat)
#                     mask_hr_init = self.kernel_normalizer(mask_hr_hr_feat, self.highpass_kernel,hamming=self.hamming_highpass)
#                     compressed_hr_feat = compressed_hr_feat + compressed_hr_feat - carafe(compressed_hr_feat,mask_hr_init,self.highpass_kernel,self.up_group, 1)

#                     mask_lr_hr_feat = self.content_encoder(compressed_hr_feat)
#                     mask_lr_init = self.kernel_normalizer(mask_lr_hr_feat, self.lowpass_kernel,hamming=self.hamming_lowpass)

#                     mask_lr_lr_feat_lr = self.content_encoder(compressed_lr_feat)
#                     mask_lr_lr_feat = F.interpolate(
#                         carafe(mask_lr_lr_feat_lr, mask_lr_init, self.lowpass_kernel, self.up_group, 2),
#                         size=compressed_hr_feat.shape[-2:], mode='nearest')
#                     mask_lr = mask_lr_hr_feat + mask_lr_lr_feat

#                     mask_lr_init = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)
#                     mask_hr_lr_feat = F.interpolate(
#                         carafe(self.content_encoder2(compressed_lr_feat), mask_lr_init, self.lowpass_kernel,
#                                self.up_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest')
#                     mask_hr = mask_hr_hr_feat + mask_hr_lr_feat
#                 else:
#                     raise NotImplementedError
#             else:
#                 mask_lr = self.content_encoder(compressed_hr_feat) + F.interpolate(
#                     self.content_encoder(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode='nearest')
#                 if self.use_high_pass:
#                     mask_hr = self.content_encoder2(compressed_hr_feat) + F.interpolate(
#                         self.content_encoder2(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode='nearest')
#         else:
#             compressed_x = F.interpolate(compressed_lr_feat, size=compressed_hr_feat.shape[-2:],
#                                          mode='nearest') + compressed_hr_feat
#             mask_lr = self.content_encoder(compressed_x)
#             if self.use_high_pass:
#                 mask_hr = self.content_encoder2(compressed_x)

#         mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)
#         if self.semi_conv:
#             lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 2)
#         else:
#             lr_feat = resize(
#                 input=lr_feat,
#                 size=hr_feat.shape[2:],
#                 mode=self.upsample_mode,
#                 align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
#             lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 1)

#         if self.use_high_pass:
#             mask_hr = self.kernel_normalizer(mask_hr, self.highpass_kernel, hamming=self.hamming_highpass)
#             hr_feat_hf = hr_feat - carafe(hr_feat, mask_hr, self.highpass_kernel, self.up_group, 1)
#             if self.hr_residual:
#                 # print('using hr_residual')
#                 hr_feat = hr_feat_hf + hr_feat
#             else:
#                 hr_feat = hr_feat_hf

#         if self.feature_resample:
#             # print(lr_feat.shape)
#             lr_feat = self.dysampler(hr_x=compressed_hr_feat,
#                                      lr_x=compressed_lr_feat, feat2sample=lr_feat)

#         # return mask_lr, hr_feat, lr_feat
#         return hr_feat + lr_feat

# class LocalSimGuidedSampler(nn.Module):
#     """
#     offset generator in FreqFusion
#     """

#     def __init__(self, in_channels, scale=2, style='lp', groups=4, use_direct_scale=True, kernel_size=1, local_window=3,
#                  sim_type='cos', norm=True, direction_feat='sim_concat'):
#         super().__init__()
#         assert scale == 2
#         assert style == 'lp'

#         self.scale = scale
#         self.style = style
#         self.groups = groups
#         self.local_window = local_window
#         self.sim_type = sim_type
#         self.direction_feat = direction_feat

#         if style == 'pl':
#             assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
#         assert in_channels >= groups and in_channels % groups == 0

#         if style == 'pl':
#             in_channels = in_channels // scale ** 2
#             out_channels = 2 * groups
#         else:
#             out_channels = 2 * groups * scale ** 2
#         if self.direction_feat == 'sim':
#             self.offset = nn.Conv2d(local_window ** 2 - 1, out_channels, kernel_size=kernel_size,
#                                     padding=kernel_size // 2)
#         elif self.direction_feat == 'sim_concat':
#             self.offset = nn.Conv2d(in_channels + local_window ** 2 - 1, out_channels, kernel_size=kernel_size,
#                                     padding=kernel_size // 2)
#         else:
#             raise NotImplementedError
#         normal_init(self.offset, std=0.001)
#         if use_direct_scale:
#             if self.direction_feat == 'sim':
#                 self.direct_scale = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
#                                               padding=kernel_size // 2)
#             elif self.direction_feat == 'sim_concat':
#                 self.direct_scale = nn.Conv2d(in_channels + local_window ** 2 - 1, out_channels,
#                                               kernel_size=kernel_size, padding=kernel_size // 2)
#             else:
#                 raise NotImplementedError
#             constant_init(self.direct_scale, val=0.)

#         out_channels = 2 * groups
#         if self.direction_feat == 'sim':
#             self.hr_offset = nn.Conv2d(local_window ** 2 - 1, out_channels, kernel_size=kernel_size,
#                                        padding=kernel_size // 2)
#         elif self.direction_feat == 'sim_concat':
#             self.hr_offset = nn.Conv2d(in_channels + local_window ** 2 - 1, out_channels, kernel_size=kernel_size,
#                                        padding=kernel_size // 2)
#         else:
#             raise NotImplementedError
#         normal_init(self.hr_offset, std=0.001)

#         if use_direct_scale:
#             if self.direction_feat == 'sim':
#                 self.hr_direct_scale = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
#                                                  padding=kernel_size // 2)
#             elif self.direction_feat == 'sim_concat':
#                 self.hr_direct_scale = nn.Conv2d(in_channels + local_window ** 2 - 1, out_channels,
#                                                  kernel_size=kernel_size, padding=kernel_size // 2)
#             else:
#                 raise NotImplementedError
#             constant_init(self.hr_direct_scale, val=0.)

#         self.norm = norm
#         if self.norm:
#             self.norm_hr = nn.GroupNorm(in_channels // 8, in_channels)
#             self.norm_lr = nn.GroupNorm(in_channels // 8, in_channels)
#         else:
#             self.norm_hr = nn.Identity()
#             self.norm_lr = nn.Identity()
#         self.register_buffer('init_pos', self._init_pos())

#     def _init_pos(self):
#         h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
#         return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

#     def sample(self, x, offset, scale=None):
#         if scale is None: scale = self.scale
#         B, _, H, W = offset.shape
#         offset = offset.view(B, 2, -1, H, W)
#         coords_h = torch.arange(H) + 0.5
#         coords_w = torch.arange(W) + 0.5
#         coords = torch.stack(torch.meshgrid([coords_w, coords_h])
#                              ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
#         normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
#         coords = 2 * (coords + offset) / normalizer - 1
#         coords = F.pixel_shuffle(coords.view(B, -1, H, W), scale).view(
#             B, 2, -1, scale * H, scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
#         return F.grid_sample(x.reshape(B * self.groups, -1, x.size(-2), x.size(-1)), coords, mode='bilinear',
#                              align_corners=False, padding_mode="border").view(B, -1, scale * H, scale * W)

#     def forward(self, hr_x, lr_x, feat2sample):
#         hr_x = self.norm_hr(hr_x)
#         lr_x = self.norm_lr(lr_x)

#         if self.direction_feat == 'sim':
#             hr_sim = compute_similarity(hr_x, self.local_window, dilation=2, sim='cos')
#             lr_sim = compute_similarity(lr_x, self.local_window, dilation=2, sim='cos')
#         elif self.direction_feat == 'sim_concat':
#             hr_sim = torch.cat([hr_x, compute_similarity(hr_x, self.local_window, dilation=2, sim='cos')], dim=1)
#             lr_sim = torch.cat([lr_x, compute_similarity(lr_x, self.local_window, dilation=2, sim='cos')], dim=1)
#             hr_x, lr_x = hr_sim, lr_sim
#         # offset = self.get_offset(hr_x, lr_x)
#         offset = self.get_offset_lp(hr_x, lr_x, hr_sim, lr_sim)
#         return self.sample(feat2sample, offset)

#     # def get_offset_lp(self, hr_x, lr_x):
#     def get_offset_lp(self, hr_x, lr_x, hr_sim, lr_sim):
#         if hasattr(self, 'direct_scale'):
#             # offset = (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * (self.direct_scale(lr_x) + F.pixel_unshuffle(self.hr_direct_scale(hr_x), self.scale)).sigmoid() + self.init_pos
#             offset = (self.offset(lr_sim) + F.pixel_unshuffle(self.hr_offset(hr_sim), self.scale)) * (
#                         self.direct_scale(lr_x) + F.pixel_unshuffle(self.hr_direct_scale(hr_x),
#                                                                     self.scale)).sigmoid() + self.init_pos
#             # offset = (self.offset(lr_sim) + F.pixel_unshuffle(self.hr_offset(hr_sim), self.scale)) * (self.direct_scale(lr_sim) + F.pixel_unshuffle(self.hr_direct_scale(hr_sim), self.scale)).sigmoid() + self.init_pos
#         else:
#             offset = (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * 0.25 + self.init_pos
#         return offset

#     def get_offset(self, hr_x, lr_x):
#         if self.style == 'pl':
#             raise NotImplementedError
#         return self.get_offset_lp(hr_x, lr_x)
# def compute_similarity(input_tensor, k=3, dilation=1, sim='cos'):
#     """
#     计算输入张量中每一点与周围KxK范围内的点的余弦相似度。

#     参数：
#     - input_tensor: 输入张量，形状为[B, C, H, W]
#     - k: 范围大小，表示周围KxK范围内的点

#     返回：
#     - 输出张量，形状为[B, KxK-1, H, W]
#     """
#     B, C, H, W = input_tensor.shape
#     # 使用零填充来处理边界情况
#     # padded_input = F.pad(input_tensor, (k // 2, k // 2, k // 2, k // 2), mode='constant', value=0)

#     # 展平输入张量中每个点及其周围KxK范围内的点
#     unfold_tensor = F.unfold(input_tensor, k, padding=(k // 2) * dilation, dilation=dilation)  # B, CxKxK, HW
#     # print(unfold_tensor.shape)
#     unfold_tensor = unfold_tensor.reshape(B, C, k ** 2, H, W)

#     # 计算余弦相似度
#     if sim == 'cos':
#         similarity = F.cosine_similarity(unfold_tensor[:, :, k * k // 2:k * k // 2 + 1], unfold_tensor[:, :, :], dim=1)
#     elif sim == 'dot':
#         similarity = unfold_tensor[:, :, k * k // 2:k * k // 2 + 1] * unfold_tensor[:, :, :]
#         similarity = similarity.sum(dim=1)
#     else:
#         raise NotImplementedError

#     # 移除中心点的余弦相似度，得到[KxK-1]的结果
#     similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1)

#     # 将结果重塑回[B, KxK-1, H, W]的形状
#     similarity = similarity.view(B, k * k - 1, H, W)
#     return similarity

# # if __name__ == "__main__":
# #     # 假设 HR 和 LR 特征图的输入
# #     hr_feat = torch.randn(1, 64, 128, 128).cuda()
# #     lr_feat = torch.randn(1, 64, 64, 64).cuda()

# #     # 创建 FreqFusion 模块实例
# #     freq_fusion = FreqFusion(
# #         hr_channels=64,  # 高分辨率通道数
# #         lr_channels=64,  # 低分辨率通道数
# #     ).cuda()

# #     output = freq_fusion(hr_feat, lr_feat)

# #     print("hr_feat size:",hr_feat.shape)
# #     print("output size:", output.shape)

# import os
# import torch
# import numpy as np
# from torch import nn
# import warnings

# warnings.filterwarnings("ignore")

# """
# This code is mainly the deformation process of our DSConv
# """


# class DSConv(nn.Module):

#     def __init__(self, in_ch, out_ch, kernel_size, extend_scope, morph, if_offset, device):
#         """
#         The Dynamic Snake Convolution
#         :param in_ch: input channel
#         :param out_ch: output channel
#         :param kernel_size: the size of kernel
#         :param extend_scope: the range to expand (default 1 for this method)
#         :param morph: the morphology of the convolution kernel is mainly divided into two types
#                         along the x-axis (0) and the y-axis (1) (see the paper for details)
#         :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
#         :param device: set on gpu
#         """
#         super(DSConv, self).__init__()
#         self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
#         self.bn = nn.BatchNorm2d(2 * kernel_size)
#         self.kernel_size = kernel_size

#         # two types of the DSConv (along x-axis and y-axis)
#         self.dsc_conv_x = nn.Conv2d(
#             in_ch,
#             out_ch,
#             kernel_size=(kernel_size, 1),
#             stride=(kernel_size, 1),
#             padding=0,
#         )
#         self.dsc_conv_y = nn.Conv2d(
#             in_ch,
#             out_ch,
#             kernel_size=(1, kernel_size),
#             stride=(1, kernel_size),
#             padding=0,
#         )

#         self.gn = nn.GroupNorm(out_ch // 4, out_ch)
#         self.relu = nn.ReLU(inplace=True)

#         self.extend_scope = extend_scope
#         self.morph = morph
#         self.if_offset = if_offset
#         self.device = device

#     def forward(self, f):
#         offset = self.offset_conv(f)
#         offset = self.bn(offset)
#         # We need a range of deformation between -1 and 1 to mimic the snake's swing
#         offset = torch.tanh(offset)
#         input_shape = f.shape
#         dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph, self.device)
#         deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
#         if self.morph == 0:
#             x = self.dsc_conv_x(deformed_feature)
#             x = self.gn(x)
#             x = self.relu(x)
#             return x
#         else:
#             x = self.dsc_conv_y(deformed_feature)
#             x = self.gn(x)
#             x = self.relu(x)
#             return x


# # Core code, for ease of understanding, we mark the dimensions of input and output next to the code
# class DSC(object):

#     def __init__(self, input_shape, kernel_size, extend_scope, morph, device):
#         self.num_points = kernel_size
#         self.width = input_shape[2]
#         self.height = input_shape[3]
#         self.morph = morph
#         self.device = device
#         self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope

#         # define feature map shape
#         """
#         B: Batch size  C: Channel  W: Width  H: Height
#         """
#         self.num_batch = input_shape[0]
#         self.num_channels = input_shape[1]

#     """
#     input: offset [B,2*K,W,H]  K: Kernel size (2*K: 2D image, deformation contains <x_offset> and <y_offset>)
#     output_x: [B,1,W,K*H]   coordinate map
#     output_y: [B,1,K*W,H]   coordinate map
#     """

#     def _coordinate_map_3D(self, offset, if_offset):
#         # offset
#         y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

#         y_center = torch.arange(0, self.width).repeat([self.height])
#         y_center = y_center.reshape(self.height, self.width)
#         y_center = y_center.permute(1, 0)
#         y_center = y_center.reshape([-1, self.width, self.height])
#         y_center = y_center.repeat([self.num_points, 1, 1]).float()
#         y_center = y_center.unsqueeze(0)

#         x_center = torch.arange(0, self.height).repeat([self.width])
#         x_center = x_center.reshape(self.width, self.height)
#         x_center = x_center.permute(0, 1)
#         x_center = x_center.reshape([-1, self.width, self.height])
#         x_center = x_center.repeat([self.num_points, 1, 1]).float()
#         x_center = x_center.unsqueeze(0)

#         if self.morph == 0:
#             """
#             Initialize the kernel and flatten the kernel
#                 y: only need 0
#                 x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
#                 !!! The related PPT will be submitted later, and the PPT will contain the whole changes of each step
#             """
#             y = torch.linspace(0, 0, 1)
#             x = torch.linspace(
#                 -int(self.num_points // 2),
#                 int(self.num_points // 2),
#                 int(self.num_points),
#             )

#             y, x = torch.meshgrid(y, x)
#             y_spread = y.reshape(-1, 1)
#             x_spread = x.reshape(-1, 1)

#             y_grid = y_spread.repeat([1, self.width * self.height])
#             y_grid = y_grid.reshape([self.num_points, self.width, self.height])
#             y_grid = y_grid.unsqueeze(0)  # [B*K*K, W,H]

#             x_grid = x_spread.repeat([1, self.width * self.height])
#             x_grid = x_grid.reshape([self.num_points, self.width, self.height])
#             x_grid = x_grid.unsqueeze(0)  # [B*K*K, W,H]

#             y_new = y_center + y_grid
#             x_new = x_center + x_grid

#             y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(self.device)
#             x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(self.device)

#             y_offset_new = y_offset.detach().clone()

#             if if_offset:
#                 y_offset = y_offset.permute(1, 0, 2, 3)
#                 y_offset_new = y_offset_new.permute(1, 0, 2, 3)
#                 center = int(self.num_points // 2)

#                 # The center position remains unchanged and the rest of the positions begin to swing
#                 # This part is quite simple. The main idea is that "offset is an iterative process"
#                 y_offset_new[center] = 0
#                 for index in range(1, center):
#                     y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
#                     y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
#                 y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(self.device)
#                 y_new = y_new.add(y_offset_new.mul(self.extend_scope))

#             y_new = y_new.reshape(
#                 [self.num_batch, self.num_points, 1, self.width, self.height])
#             y_new = y_new.permute(0, 3, 1, 4, 2)
#             y_new = y_new.reshape([
#                 self.num_batch, self.num_points * self.width, 1 * self.height
#             ])
#             x_new = x_new.reshape(
#                 [self.num_batch, self.num_points, 1, self.width, self.height])
#             x_new = x_new.permute(0, 3, 1, 4, 2)
#             x_new = x_new.reshape([
#                 self.num_batch, self.num_points * self.width, 1 * self.height
#             ])
#             return y_new, x_new

#         else:
#             """
#             Initialize the kernel and flatten the kernel
#                 y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
#                 x: only need 0
#             """
#             y = torch.linspace(
#                 -int(self.num_points // 2),
#                 int(self.num_points // 2),
#                 int(self.num_points),
#             )
#             x = torch.linspace(0, 0, 1)

#             y, x = torch.meshgrid(y, x)
#             y_spread = y.reshape(-1, 1)
#             x_spread = x.reshape(-1, 1)

#             y_grid = y_spread.repeat([1, self.width * self.height])
#             y_grid = y_grid.reshape([self.num_points, self.width, self.height])
#             y_grid = y_grid.unsqueeze(0)

#             x_grid = x_spread.repeat([1, self.width * self.height])
#             x_grid = x_grid.reshape([self.num_points, self.width, self.height])
#             x_grid = x_grid.unsqueeze(0)

#             y_new = y_center + y_grid
#             x_new = x_center + x_grid

#             y_new = y_new.repeat(self.num_batch, 1, 1, 1)
#             x_new = x_new.repeat(self.num_batch, 1, 1, 1)

#             y_new = y_new.to(self.device)
#             x_new = x_new.to(self.device)
#             x_offset_new = x_offset.detach().clone()

#             if if_offset:
#                 x_offset = x_offset.permute(1, 0, 2, 3)
#                 x_offset_new = x_offset_new.permute(1, 0, 2, 3)
#                 center = int(self.num_points // 2)
#                 x_offset_new[center] = 0
#                 for index in range(1, center):
#                     x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
#                     x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
#                 x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(self.device)
#                 x_new = x_new.add(x_offset_new.mul(self.extend_scope))

#             y_new = y_new.reshape(
#                 [self.num_batch, 1, self.num_points, self.width, self.height])
#             y_new = y_new.permute(0, 3, 1, 4, 2)
#             y_new = y_new.reshape([
#                 self.num_batch, 1 * self.width, self.num_points * self.height
#             ])
#             x_new = x_new.reshape(
#                 [self.num_batch, 1, self.num_points, self.width, self.height])
#             x_new = x_new.permute(0, 3, 1, 4, 2)
#             x_new = x_new.reshape([
#                 self.num_batch, 1 * self.width, self.num_points * self.height
#             ])
#             return y_new, x_new

#     """
#     input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H] 
#     output: [N,1,K*D,K*W,K*H]  deformed feature map
#     """

#     def _bilinear_interpolate_3D(self, input_feature, y, x):
#         y = y.reshape([-1]).float()
#         x = x.reshape([-1]).float()

#         zero = torch.zeros([]).int()
#         max_y = self.width - 1
#         max_x = self.height - 1

#         # find 8 grid locations
#         y0 = torch.floor(y).int()
#         y1 = y0 + 1
#         x0 = torch.floor(x).int()
#         x1 = x0 + 1

#         # clip out coordinates exceeding feature map volume
#         y0 = torch.clamp(y0, zero, max_y)
#         y1 = torch.clamp(y1, zero, max_y)
#         x0 = torch.clamp(x0, zero, max_x)
#         x1 = torch.clamp(x1, zero, max_x)

#         input_feature_flat = input_feature.flatten()
#         input_feature_flat = input_feature_flat.reshape(
#             self.num_batch, self.num_channels, self.width, self.height)
#         input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
#         input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
#         dimension = self.height * self.width

#         base = torch.arange(self.num_batch) * dimension
#         base = base.reshape([-1, 1]).float()

#         repeat = torch.ones([self.num_points * self.width * self.height
#                              ]).unsqueeze(0)
#         repeat = repeat.float()

#         base = torch.matmul(base, repeat)
#         base = base.reshape([-1])

#         base = base.to(self.device)

#         base_y0 = base + y0 * self.height
#         base_y1 = base + y1 * self.height

#         # top rectangle of the neighbourhood volume
#         index_a0 = base_y0 - base + x0
#         index_c0 = base_y0 - base + x1

#         # bottom rectangle of the neighbourhood volume
#         index_a1 = base_y1 - base + x0
#         index_c1 = base_y1 - base + x1

#         # get 8 grid values
#         value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(self.device)
#         value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(self.device)
#         value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(self.device)
#         value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(self.device)

#         # find 8 grid locations
#         y0 = torch.floor(y).int()
#         y1 = y0 + 1
#         x0 = torch.floor(x).int()
#         x1 = x0 + 1

#         # clip out coordinates exceeding feature map volume
#         y0 = torch.clamp(y0, zero, max_y + 1)
#         y1 = torch.clamp(y1, zero, max_y + 1)
#         x0 = torch.clamp(x0, zero, max_x + 1)
#         x1 = torch.clamp(x1, zero, max_x + 1)

#         x0_float = x0.float()
#         x1_float = x1.float()
#         y0_float = y0.float()
#         y1_float = y1.float()

#         vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(self.device)
#         vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(self.device)
#         vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(self.device)
#         vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(self.device)

#         outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 +
#                    value_c1 * vol_c1)

#         if self.morph == 0:
#             outputs = outputs.reshape([
#                 self.num_batch,
#                 self.num_points * self.width,
#                 1 * self.height,
#                 self.num_channels,
#             ])
#             outputs = outputs.permute(0, 3, 1, 2)
#         else:
#             outputs = outputs.reshape([
#                 self.num_batch,
#                 1 * self.width,
#                 self.num_points * self.height,
#                 self.num_channels,
#             ])
#             outputs = outputs.permute(0, 3, 1, 2)
#         return outputs

#     def deform_conv(self, input, offset, if_offset):
#         y, x = self._coordinate_map_3D(offset, if_offset)
#         deformed_feature = self._bilinear_interpolate_3D(input, y, x)
#         return deformed_feature


# # if __name__ == '__main__':
# #     os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     A = np.random.rand(2, 32, 256, 256)
# #     # A = np.ones(shape=(3, 2, 2, 3), dtype=np.float32)
# #     # print(A)
# #     A = A.astype(dtype=np.float32)
# #     A = torch.from_numpy(A)
# #     # print(A.shape)
# #     conv0 = DSConv(
# #         in_ch=32,
# #         out_ch=32,
# #         kernel_size=15,
# #         extend_scope=1,
# #         morph=0,
# #         if_offset=True,
# #         device=device)
# #     if torch.cuda.is_available():
# #         A = A.to(device)
# #         conv0 = conv0.to(device)
# #     out = conv0(A)
# #     print(out.shape)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# '''
# 来自 AAAI 2025顶会
# 即插即用卷积模块: PSConv 风车形状卷积模块  Pinwheel-shaped Convolution（PConv）
# 小目标检测损失函数: SDIoU,作为YOLOv8v10v11小目标检测任务的损失函数改进点！有效涨点！

# 近年来，基于卷积神经网络 （CNN） 的红外小目标检测方法取得了出色的性能。
# 然而，这些方法通常采用标准卷积，而忽略了红外小目标像素分布的空间特性。
# 因此，我们提出了一种新的风车形卷积 （PConv） 来替代骨干网络下层的标准卷积。
# PConv 更好地与暗淡小目标的像素高斯空间分布保持一致，增强了特征提取，显著增加了感受野，并且仅引入了最小的参数增加。
# 此外，虽然最近的损失函数结合了尺度和位置损失，但它们没有充分考虑这些损失在不同目标尺度上的不同灵敏度，从而限制了对暗小目标的检测性能。
# 为了克服这个问题，我们提出了一种基于尺度的动态 （SD） 损失，它根据目标大小动态调整尺度和位置损失的影响，从而提高网络检测不同尺度目标的能力。
# 我们构建了一个新的基准 SIRST-UAVB，这是迄今为止最大、最具挑战性的实拍单帧红外小目标检测数据集。
# 最后，通过将 PConv 和 SD Loss 集成到最新的小目标检测算法中，
# 我们在 IRSTD-1K 和 SIRST-UAVB 数据集上实现了显著的性能改进，验证了我们方法的有效性和通用性。

# 适用于：红外小目标检测，小目标检测任务，目标检测，图像分割，语义分割，图像增强等所有一切计算机视觉CV任务通用的即插即用卷积模块。
# '''
# class BiasFree_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(BiasFree_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)

#         assert len(normalized_shape) == 1

#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.normalized_shape = normalized_shape

#     def forward(self, x):
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return x / torch.sqrt(sigma+1e-5) * self.weight


# class WithBias_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(WithBias_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)

#         assert len(normalized_shape) == 1

#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.normalized_shape = normalized_shape

#     def forward(self, x):
#         mu = x.mean(-1, keepdim=True)
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

# class LayerNorm(nn.Module):
#     def __init__(self, dim, LayerNorm_type):
#         super(LayerNorm, self).__init__()
#         if LayerNorm_type == 'BiasFree':
#             self.body = BiasFree_LayerNorm(dim)
#         else:
#             self.body = WithBias_LayerNorm(dim)

#     def forward(self, x):
#         h, w = x.shape[-2:]
#         return to_4d(self.body(to_3d(x)), h, w)

# def autopad(k, p=None, d=1):  # kernel, padding, dilation
#     """Pad to 'same' shape outputs."""
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p


# class Conv(nn.Module):
#     """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

#     default_act = nn.LeakyReLU(0.1, inplace=True)  # default activation

#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         """Initialize Conv layer with given arguments including activation."""
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         # self.bn = nn.BatchNorm2d(c2)
#         self.bn = LayerNorm(c2, 'BiasFree')
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         """Apply convolution, batch normalization and activation to input tensor."""
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         """Perform transposed convolution of 2D data."""
#         return self.act(self.conv(x))




# from functools import partial
# import math
# '''
# 来自CVPR 2025 顶会
# 即插即用模块： SCM 特征位移混合模块
# 带来两个二次创新模块 ： SPConv移动风车卷积模块  ； SCEU 移动有效上采样模块

# 主要内容：
# 模型二值化在实现卷积神经网络（CNN）的实时高效计算方面取得了显著进展，为视觉Transformer（ViT）在边缘设备上的部署挑战提供了潜在解决方案。
# 然而，由于CNN和Transformer架构的结构差异，直接将二值化CNN策略应用于ViT模型会导致性能显著下降。
# 为解决这一问题，我们提出了BHViT——一种适合二值化的混合ViT架构及其全二值化模型，其设计基于以下三个重要观察：

# 1.局部信息交互与分层特征聚合：BHViT利用从粗到细的分层特征聚合技术，减少因冗余token带来的计算开销。
# 2.基于移位操作的新型模块：提出一种基于移位操作的模块（SCM），在不显著增加计算负担的情况下提升二值化多层感知机（MLP）的性能。
# 3.量化分解的注意力矩阵二值化方法：提出一种基于量化分解的创新方法，用于评估二值化注意力矩阵中各token的重要性。

# 该Shift_channel_mix（SCM）模块是论文中提出的一个轻量化模块，用于增强二进制多层感知器（MLP）在二进制视觉变换器（BViT）中的表现。
# 它通过对输入特征图进行不同的移位操作，帮助缓解信息丢失和梯度消失的问题，从而提高网络的性能，同时避免增加过多的计算开销。
# SCM模块的主要操作包括：
# 1.水平移位（Horizontal Shift）：通过torch.roll函数将特征图的列按指定的大小进行右/左移操作。这种操作模拟了在处理二进制向量时的特征循环，增强了表示能力。
# 2.垂直移位（Vertical Shift）：类似于水平移位，垂直移位会使特征图的行发生上下移动。这有助于捕获跨行的信息，同时适应不同的特征维度。
# 在代码实现中，torch.chunk将输入特征图沿着通道维度分成四个部分，之后通过不同的移位操作处理每一部分，最后将处理后的四个部分通过torch.cat拼接起来，形成最终的输出。

# SCM模块适合：目标检测，图像分割，语义分割，图像增强，图像去噪，遥感语义分割，图像分类等所有CV任务通用的即插即用模块
# 这个SCM轻量小巧模块，建议最好搭配其它模块一起使用！

# '''

# class Shift_channel_mix(nn.Module):
#     def __init__(self, shift_size=1):
#         super(Shift_channel_mix, self).__init__()
#         self.shift_size = shift_size

#     def forward(self, x):  # x的张量 [B,C,H,W]
#         x1, x2, x3, x4 = x.chunk(4, dim=1)

#         x1 = torch.roll(x1, self.shift_size, dims=2)  # [:,:,1:,:]

#         x2 = torch.roll(x2, -self.shift_size, dims=2)  # [:,:,:-1,:]

#         x3 = torch.roll(x3, self.shift_size, dims=3)  # [:,:,:,1:]

#         x4 = torch.roll(x4, -self.shift_size, dims=3)  # [:,:,:,:-1]

#         x = torch.cat([x1, x2, x3, x4], 1)

#         return x


# #二次创新模块 SPConv  移动风车形状卷积
# class SPConv(nn.Module):
#     ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''

#     def __init__(self, c1, c2, k=3, s=1):
#         super().__init__()

#         # 定义4种非对称填充方式，用于风车形状卷积的实现
#         p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]  # 每个元组表示 (左, 上, 右, 下) 填充
#         self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]  # 创建4个填充层

#         # 定义水平方向卷积操作，卷积核大小为 (1, k)，步幅为 s，输出通道数为 c2 // 4
#         self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)

#         # 定义垂直方向卷积操作，卷积核大小为 (k, 1)，步幅为 s，输出通道数为 c2 // 4
#         self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)

#         # 最终合并卷积结果的卷积层，卷积核大小为 (2, 2)，输出通道数为 c2
#         self.cat = Conv(c2, c2, 2, s=1, p=0)

#         self.shift_size = 1
#     def forward(self, x):
#         # 对输入 x 进行不同填充和卷积操作，得到四个方向的特征
#         yw0 = self.cw(self.pad[0](x))  # 水平方向，第一个填充方式
#         yw1 = self.cw(self.pad[1](x))  # 水平方向，第二个填充方式
#         yh0 = self.ch(self.pad[2](x))  # 垂直方向，第一个填充方式
#         yh1 = self.ch(self.pad[3](x))  # 垂直方向，第二个填充方式

#         x1 = torch.roll(yw0, self.shift_size, dims=2)  # [:,:,1:,:]

#         x2 = torch.roll(yw1, -self.shift_size, dims=2)  # [:,:,:-1,:]

#         x3 = torch.roll(yh0, self.shift_size, dims=3)  # [:,:,:,1:]

#         x4 = torch.roll(yh1, -self.shift_size, dims=3)  # [:,:,:,:-1]

#         out = torch.cat([x1, x2, x3, x4], 1)
#         # 将四个卷积结果在通道维度拼接，并通过一个额外的卷积层处理，最终输出
#         return self.cat(out)  # 在通道维度拼接，并通过 cat 卷积层处理

# # 输入 B C H W, 输出 B C H W
# # if __name__ == "__main__":
# #     input = torch.randn(1,32,64, 64)  # 创建一个形状为 (1,32,64, 64)
# #     SCM = Shift_channel_mix()
# #     output = SCM(input)  # 通过SCM模块计算输出
# #     print('SCM_Input size:', input.size())  # 打印输入张量的形状
# #     print('SCM_Output size:', output.size())  # 打印输出张量的形状

# #     input = torch.randn(1, 32, 64, 64)  # 创建一个形状为 (1,32,64, 64)
# #     SPConv = SPConv(32,32) #二次创新SPConv卷积模块
# #     output = SPConv(input)
# #     print('二次创新SPConv_Input size:', input.size())  # 打印输入张量的形状
# #     print('二次创新SPConv_Output size:', output.size())  # 打印输出张量的形状
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange

# '''
# 二次创新模块：CGAFusion（2024 TIP顶刊） 结合 CAFM（2024 SCI 二区）：CAFMFusion用于高频与低频特征校准/融合模块 （冲二，三，保四）

# CAFM:所提出的卷积和注意力特征融合模块。它由局部和全局分支组成。
#     在局部分支中，采用卷积和通道洗牌进行局部特征提取。
#     在全局分支中，注意力机制用于对长程特征依赖关系进行建模。

# CGAFusion（2024 TIP顶刊）:我们提出了一种新的注意机制，可以强调用特征编码的更多有用的信息，以有效地提高性能。
#     此外，还提出了一种基于CGA的混合融合方案，可以有效地将编码器部分的低级特征与相应的高级特征进行融合。

# 强强联手：CGAFusion（2024 TIP顶刊） 结合 CAFM（2024 SCI 二区）：CAFMFusion
#         CAFMFusion用于低级特征与高级特征校准/融合模块 （冲二，三，保四）
# 适用于：图像去噪，图像增强，目标检测，语义分割，实例分割，图像恢复，暗光增强等所有CV2维任务
# '''
# def to_3d(x):
#     return rearrange(x, 'b c h w -> b (h w) c')
# def to_4d(x, h, w):
#     return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
# class PixelAttention(nn.Module):
#     def __init__(self, dim):
#         super(PixelAttention, self).__init__()
#         self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, pattn1):
#         B, C, H, W = x.shape
#         x = x.unsqueeze(dim=2)  # B, C, 1, H, W
#         pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
#         x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
#         x2 = rearrange(x2, 'b c t h w -> b (c t) h w')
#         pattn2 = self.pa2(x2)
#         pattn2 = self.sigmoid(pattn2)
#         return pattn2
# class CAFM(nn.Module):
#     def __init__(self, dim, num_heads=4, bias=False):
#         super(CAFM, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
#         self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3, bias=bias)
#         self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
#         self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)
#         self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True, groups=dim // self.num_heads, padding=1)

#     def forward(self, x):
#         b, c, h, w = x.shape
#         x = x.unsqueeze(2)
#         qkv = self.qkv_dwconv(self.qkv(x))
#         qkv = qkv.squeeze(2)
#         f_conv = qkv.permute(0, 2, 3, 1)
#         f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
#         f_all = self.fc(f_all.unsqueeze(2))
#         f_all = f_all.squeeze(2)

#         f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
#         f_conv = f_conv.unsqueeze(2)
#         out_conv = self.dep_conv(f_conv)
#         out_conv = out_conv.squeeze(2)

#         q, k, v = qkv.chunk(3, dim=1)
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)

#         out = (attn @ v)
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#         out = out.unsqueeze(2)
#         out = self.project_out(out)
#         out = out.squeeze(2)
#         output = out + out_conv
#         return output

# class CAFMFusion(nn.Module):
#     def __init__(self, dim):
#         super(CAFMFusion, self).__init__()
#         self.CAFM = CAFM(dim)
#         self.PixelAttention = PixelAttention(dim)
#         self.conv = nn.Conv2d(dim, dim, 1, bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, y):
#         initial = x + y
#         pattn1 = self.CAFM(initial)
#         pattn2 = self.sigmoid(self.PixelAttention(initial, pattn1))
#         result = initial + pattn2 * x + (1 - pattn2) * y
#         result = self.conv(result)
#         return result

# # if __name__ == '__main__':
# #     block = CAFMFusion(32)
# #     input1 = torch.rand(1, 32, 64, 64)
# #     input2 = torch.rand(1, 32, 64, 64)
# #     output = block(input1, input2)
# #     print('input1_size:', input1.size())
# #     print('input2_size:', input2.size())
# #     print('output_size:', output.size())
# from functools import partial
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# nonlinearity = partial(F.relu, inplace=True)

# '''
# SCI一区 2024 
# 即插即用特征增强模块：HIFA    所有CV任务通用模块
# HIFA模块是为了在医学图像分割任务中更好地结合局部与全局信息而设计的。
# 其核心目标是能够捕获低频信息（如全局结构和语义）以及高频信息（如局部边缘和纹理），
# 从而更好地模拟人类视觉系统处理图像的方式。

# HIFA模块的核心组成：
# 空间金字塔池化（SPP）：用于捕获多尺度的全局上下文信息。这可以帮助模型在不同尺度下理解全局语义。
# 多尺度膨胀卷积：通过不同的膨胀率提取局部上下文信息，从而捕获更多细节如边缘和纹理。

# 作用：
# HIFA模块能够有效融合局部细节和全局结构信息，提升分割模型的表现，特别是在复杂结构或不同尺度信息要求高的医学图像分割任务中。
# 通过这种全局与局部信息的增强与融合，HIFA模块可以实现更高效的特征编码与解码，提高分割精度。
# HIFA模块通过整合全局和局部的特征信息，为网络提供更丰富的上下文信息，显著提升了分割任务中的表现。

# HIFA特征增强模块适用于：医学图像分割，图像增强，目标检测，图像分类等所有CV任务通用模块

# '''

# # ############################################## HIFA_module###########################################
# def BNReLU(num_features):
#     return nn.Sequential(
#         nn.BatchNorm2d(num_features),
#         nn.ReLU()
#     )

# class SPP_inception_block(nn.Module):
#     def __init__(self, in_channels):
#         super(SPP_inception_block, self).__init__()
#         self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)  # [3, 3]
#         self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)  # [2, 2]
#         # self.pool = nn.MaxPool2d(kernel_size=[4, 4], stride=4) # [1, 1]
#         # self.pool = nn.MaxPool2d(kernel_size=[1, 1], stride=2) # [4, 4]
#         # self.pool = nn.MaxPool2d(kernel_size=[1, 1], stride=1)   # [7, 7]
#         self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
#         self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

#         self.dilate1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1)
#         self.dilate2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=3, padding=3)
#         self.dilate3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=5, padding=5)
#         self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, dilation=1, padding=0)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 if m.bias is not None:
#                     m.bias.data.zero_()

#     def forward(self, x):
#         b, c, h, w = x.size()  # [4, 256, 7, 7]
#         pool_1 = self.pool1(x).view(b, c, -1)  # [2, 256, 3, 3], [2, 256, 9]
#         # pool_1 = self.pool(x).view(b, c, -1)
#         pool_2 = self.pool2(x).view(b, c, -1)  # [2, 256, 2, 2], [2, 256, 4]
#         pool_3 = self.pool3(x).view(b, c, -1)  # [2, 256, 1, 1], [2, 256, 1]
#         pool_4 = self.pool4(x).view(b, c, -1)  # [2, 256, 1, 1], [2, 256, 1]

#         pool_cat = torch.cat([pool_1, pool_2, pool_3, pool_4], -1)  # [2, 256, 15]

#         dilate1_out = nonlinearity(self.dilate1(x))
#         dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
#         dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
#         dilate4_out = nonlinearity(
#             self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))  # self.conv1x1 is not necessary

#         cnn_out = dilate1_out + dilate2_out + dilate3_out + dilate4_out  # [2, 256, 7, 7]
#         cnn_out = cnn_out.view(b, c, -1)  # [2, 256, 49]

#         out = torch.cat([pool_cat, cnn_out], -1)  # [2, 256, 64]
#         out = out.permute(0, 2, 1)  # [2, 64, 256]

#         return out
# class NonLocal_spp_inception_block(nn.Module):
#     def __init__(self, in_channels=512, ratio=2):
#         super(NonLocal_spp_inception_block, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = in_channels
#         self.key_channels = in_channels // ratio
#         self.value_channels = in_channels // ratio

#         self.f_key = nn.Sequential(
#             nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0),
#             BNReLU(self.key_channels),
#         )

#         self.f_query = self.f_key

#         self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
#                                  kernel_size=1, stride=1, padding=0)

#         self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
#                            kernel_size=1, stride=1, padding=0)

#         self.spp_inception_v = SPP_inception_block(self.key_channels)
#         self.spp_inception_k = SPP_inception_block(self.key_channels)
#         nn.init.constant_(self.W.weight, 0)
#         nn.init.constant_(self.W.bias, 0)

#     def forward(self, x):
#         batch_size, h, w = x.size(0), x.size(2), x.size(3)  # [2, 512, 7, 7]

#         x_v = self.f_value(x)  # [2, 256, 7, 7]
#         value = self.spp_inception_v(x_v)  # [2, 64, 256]  15+49

#         query = self.f_query(x).view(batch_size, self.key_channels, -1)  # [2, 256, 7, 7], [2, 256, 49]
#         query = query.permute(0, 2, 1)  # [2, 49, 256]

#         x_k = self.f_key(x)  # [2, 256, 7, 7]
#         key = self.spp_inception_k(x_k)  # [2, 64, 256]  15+49
#         key = key.permute(0, 2, 1)  # # [2, 256, 64]

#         sim_map = torch.matmul(query, key)  # [2, 49, 64]
#         sim_map = (self.key_channels ** -.5) * sim_map
#         sim_map = F.softmax(sim_map, dim=-1)

#         context = torch.matmul(sim_map, value)  # [2, 49, 256]
#         context = context.permute(0, 2, 1).contiguous()
#         context = context.view(batch_size, self.value_channels, *x.size()[2:])  # [4, 256, 7, 7]
#         context = self.W(context)  # [4, 512, 7, 7]
#         return context

# class HIFA(nn.Module):
#     def __init__(self, in_channels=512, ratio=2, dropout=0.0):
#         super(HIFA, self).__init__()

#         self.NSIB = NonLocal_spp_inception_block(in_channels=in_channels, ratio=ratio)
#         self.conv_bn_dropout = nn.Sequential(
#             nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0),
#             BNReLU(in_channels)
#             # nn.Dropout2d(dropout)
#         )
#     def forward(self, feats):
#         att = self.NSIB(feats)
#         output = self.conv_bn_dropout(torch.cat([att, feats], 1))
#         return output
# # if __name__ == "__main__":
# #     # 创建一个简单的输入特征图
# #     input = torch.randn(1, 512, 32, 32)
# #     # 创建一个HIFA实例
# #     HIFA = HIFA(in_channels=512)
# #     # 将输入特征图传递给 HIFA模块
# #     output = HIFA(input)
# #     # 打印输入和输出的尺寸
# #     print(f"input  shape: {input.shape}")
# #     print(f"output shape: {output.shape}")


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import numbers

class Restormer_CNN_block(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(Restormer_CNN_block, self).__init__()
        self.embed=nn.Conv2d(in_dim, out_dim,kernel_size=3,stride=1, padding=1, bias=False,padding_mode="reflect")
        self.GlobalFeature = GlobalFeatureExtraction(dim=out_dim, num_heads = 8)
        self.LocalFeature = LocalFeatureExtraction(dim=out_dim)
        self.FFN=nn.Conv2d(out_dim*2, out_dim,kernel_size=3,stride=1, padding=1, bias=False,padding_mode="reflect")          
    def forward(self, x):
        x=self.embed(x)
        x1=self.GlobalFeature(x)
        x2=self.LocalFeature(x)
        out=self.FFN(torch.cat((x1,x2),1))
        return out
class GlobalFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,  
                 qkv_bias=False,):
        super(GlobalFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,out_fratures=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class LocalFeatureExtraction(nn.Module):
    def __init__(self,
                 dim=64,
                 num_blocks=1,
                 ):
        super(LocalFeatureExtraction, self).__init__()
        self.Extraction = nn.Sequential(*[ResBlock(dim,dim) for i in range(num_blocks)])
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True)
    def forward(self, x):
        return self.Extraction(x) 
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,padding_mode="reflect"),
        )
    def forward(self, x):
        out = self.conv(x)
        return out+x

class AttentionBase(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):

        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out
    
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                 in_features, 
                 out_fratures,
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias,padding_mode="reflect")

        self.project_out = nn.Conv2d(
            hidden_features, out_fratures, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x