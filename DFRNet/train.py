import torchvision
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import scipy.io as scio
import kornia
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from Fuse import Encoder,Decoder
from Loss import cc, Fusionloss

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MatchingImageDataset_GRAY(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.vi_files = os.listdir(os.path.join(root_dir, "vi"))
        self.ir_files = os.listdir(os.path.join(root_dir, "ir"))
        
        #self.ta_files = os.listdir(os.path.join(root_dir, "target"))

        assert len(self.vi_files) == len(self.ir_files), \
        "Number of images in 'vi' and 'ir' directories must match."

    def __len__(self):
        return len(self.vi_files)

    def __getitem__(self, idx):
        vi_image_path = os.path.join(self.root_dir, "vi", self.vi_files[idx])
        ir_image_path = os.path.join(self.root_dir, "ir", self.ir_files[idx])
        vi_image = Image.open(vi_image_path)
        vi_image_ycrcb = vi_image.convert('YCbCr')
        y_channel, cr_channel, cb_channel = vi_image_ycrcb.split()
        vi_image = y_channel.convert('RGB')
        ir_image = Image.open(ir_image_path).convert('RGB')
        
        if self.transform:
            vi_image = self.transform(vi_image)
            ir_image = self.transform(ir_image)
        return vi_image, ir_image#, ta_image

class Laplacian(nn.Module):
    def __init__(self):
        super(Laplacian, self).__init__()
        kernel = [[1, 1, 1],
                  [1, -8, 1],
                  [1, 1, 1]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 3, 3]
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        # Apply convolution with the Laplacian kernel
        weight = self.weight.to(device)
        laplacian = F.conv2d(x, weight, padding=1)
        return laplacian


class Laplacian_loss(nn.Module):
    def __init__(self):
        super(Laplacian_loss, self).__init__()
        self.laplacianconv = Laplacian()

    def forward(self,generate_img,image_vis,image_ir):
        
        y_laplacian            = self.laplacianconv(image_vis)
        ir_laplacian           = self.laplacianconv(image_ir)
        generate_img_laplacian = self.laplacianconv(generate_img)
        x_laplacian_joint      = torch.max(y_laplacian,ir_laplacian)
        loss_laplacian         = F.l1_loss(x_laplacian_joint,generate_img_laplacian)

        return loss_laplacian


transforms = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.Grayscale(1),
        transforms.ToTensor()
        ])
root = '/0Prior'
train_path = '/Train_result_CNN'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size=24
channel=64
epochs = 80
lr = 1e-4

Train_Image_Number=len(os.listdir(root))

Iter_per_epoch=(Train_Image_Number % batch_size!=0)+Train_Image_Number//batch_size


    #load_data
matching_dataset = MatchingImageDataset_GRAY(root_dir=root, transform=transforms)
train_loader = DataLoader(matching_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
# =============================================================================
# Models
# =============================================================================
# IR_Encoder = nn.DataParallel(IR_Encoder()).to(device)
VI_Encoder = nn.DataParallel(Encoder()).to(device)
AE_Decoder = nn.DataParallel(Decoder()).to(device)

# optimizer1 = optim.Adam(IR_Encoder.parameters(), lr = lr)
optimizer2 = optim.Adam(VI_Encoder.parameters(), lr = lr, weight_decay=0.0005)
optimizer3 = optim.Adam(AE_Decoder.parameters(), lr = lr, weight_decay=0.0005)

optim_step = 20
optim_gamma = 0.5

# # scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [epochs//3,epochs//3*2], gamma=0.1)
# scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
# scheduler3 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss()
SmoothL1Loss=nn.SmoothL1Loss()
L1Loss=nn.L1Loss()
ssim = kornia.losses.SSIMLoss(11, reduction='mean')
# L2Loss = torch.norm()
# L2Loss = torch.norm
# =============================================================================
# Training
# =============================================================================
print('============ Training Begins ===============')
loss_train=[]
mse_loss_B_train=[]
mse_loss_D_train=[]
mse_loss_VF_train=[]
mse_loss_IF_train=[]
Gradient_loss_train=[]
L2_loss_train=[]
lr_list1=[]
lr_list2=[]
lr_list3=[]
alpha_list=[]
# step = 0
for iteration in range(epochs):
    
    # IR_Encoder.train()
    VI_Encoder.train()
    AE_Decoder.train()

    for imgs_vi, imgs_ir in tqdm(train_loader):      
        data_VIS=imgs_vi.cuda()
        data_IR=imgs_ir.cuda()
        
        # optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        # =====================================================================
        # Calculate loss 
        # =====================================================================
        vi1, vi_2, vi_3, vi_4 = VI_Encoder(data_VIS)
        ir1, ir_2, ir_3, ir_4 = VI_Encoder(data_IR)

        out4, out12,img_recon_V=AE_Decoder(vi1, ir1, vi_2, ir_2, vi_3, vi_4, ir_3, ir_4, data_VIS)
        # img_recon_I=AE_Decoder(feature_I_1,feature_I_2,feature_I_B, feature_I_D)

        mse_loss_B  = 0.1 * torch.norm(vi_3-ir_3, p=2) /256 + L1Loss(vi_3, ir_3)
        mse_loss_D  = 0.1 * torch.norm(vi_4-ir_4, p=2) /256 + L1Loss(vi_4, ir_4)

        mse_loss_V = 5 * ssim(data_VIS, out12) + MSELoss(data_VIS, out12)
        mse_loss_I = 5 * ssim(data_IR, out4) + MSELoss(data_IR, out4)

        laplacian_loss = Laplacian_loss()
        mse_loss_VF = 0.5*torch.norm(torch.max(data_IR,data_VIS)-img_recon_V, p=2) + 5*laplacian_loss(img_recon_V,data_IR,data_VIS)  #+ \
                        #5.0*ssim(torch.max(data_VIS, data_VIS), img_recon_V) + 5*MSELoss(torch.max(data_IR, data_VIS), img_recon_V)
        
        fusion = Fusionloss()

        fusionloss,loss_1,loss_grad = fusion(data_IR, data_VIS, img_recon_V)   #max L1, and max gradloss
        # mse_loss_IF = 5*ssim(data_IR,  img_recon_V)+MSELoss(data_IR,  img_recon_V)

        Gradient_loss = L1Loss(
                kornia.filters.SpatialGradient()(data_VIS),
                kornia.filters.SpatialGradient()(img_recon_V)
                )
        #Total loss
        loss = mse_loss_VF + fusionloss + mse_loss_B - 0.5*mse_loss_D + mse_loss_V + mse_loss_I
        # loss = fusionloss + 0.5*mse_loss_B + 0.5*mse_loss_D + 0.2*norm_loss + mse_loss_VF
   
        loss.backward()
        # optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        
        los = loss.item()
        los_B = mse_loss_B.item()
        los_D = mse_loss_D.item()
        los_VF = mse_loss_VF.item()
        # los_IF = mse_loss_IF.item()
        los_G = loss_grad.item()
        
        print('Epoch/step: %d/%d, loss: %.7f, lr: %f' %(iteration+1, 1, los, optimizer2.state_dict()['param_groups'][0]['lr']))

        #Save Loss
        loss_train.append(loss.item())
        mse_loss_B_train.append(mse_loss_B.item())
        mse_loss_D_train.append(mse_loss_D.item())
        mse_loss_VF_train.append(mse_loss_VF.item())
        # mse_loss_IF_train.append(mse_loss_IF.item())
        Gradient_loss_train.append(loss_grad.item())
    # scheduler1.step()
    scheduler2.step()
    scheduler3.step()
    # lr_list1.append(optimizer1.state_dict()['param_groups'][0]['lr'])
    lr_list2.append(optimizer2.state_dict()['param_groups'][0]['lr'])
    lr_list3.append(optimizer3.state_dict()['param_groups'][0]['lr'])

if not os.path.exists(train_path):
    os.makedirs(train_path, exist_ok=False)

# # Save Weights and result
# torch.save( {'weight': IR_Encoder.state_dict(), 'epoch':epochs}, 
#    os.path.join(train_path,'IRcoder_weight.pkl'))
torch.save( {'weight': VI_Encoder.state_dict(), 'epoch':epochs}, 
   os.path.join(train_path,'VIcoder_weight.pkl'))
torch.save( {'weight': AE_Decoder.state_dict(), 'epoch':epochs}, 
   os.path.join(train_path,'Decoder_weight.pkl'))
# plot
def Average_loss(loss):
    return [sum(loss[i*Iter_per_epoch:(i+1)*Iter_per_epoch])/Iter_per_epoch for i in range(int(len(loss)/Iter_per_epoch))]

plt.figure(figsize=[12,8])
plt.subplot(2,3,1), plt.plot(Average_loss(loss_train)), plt.title('Loss')
plt.subplot(2,3,2), plt.plot(Average_loss(mse_loss_B_train)), plt.title('Base_layer_loss')
plt.subplot(2,3,3), plt.plot(Average_loss(mse_loss_D_train)), plt.title('Detail_layer_loss')
plt.subplot(2,3,4), plt.plot(Average_loss(mse_loss_VF_train)), plt.title('V_recon_loss')
plt.subplot(2,3,5), plt.plot(Average_loss(mse_loss_IF_train)), plt.title('I_recon_loss')
plt.subplot(2,3,6), plt.plot(Average_loss(Gradient_loss_train)), plt.title('Gradient_loss')
plt.tight_layout()
plt.savefig(os.path.join(train_path,'curve_per_epoch.png'))