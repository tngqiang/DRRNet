# -*- coding: utf-8 -*-
"""
@author: Zixiang Zhao (zixiangzhao@stu.xjtu.edu.cn)

Pytorch implement for "DIDFuse: Deep Image Decomposition for Infrared and Visible Image Fusion" (IJCAI 2020)

https://www.ijcai.org/Proceedings/2020/135
"""

import numpy as np
import torch
import os
from PIL import Image
from skimage.io import imsave
from utils_didfuse import Test_fusion
import cv2
from Evaluator import Evaluator

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
# GPU_number = os.environ['CUDA_VISIBLE_DEVICES']


def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def img_save(image,imagename,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Gray_pic
    image = image.astype(np.uint8)
    imsave(os.path.join(savepath, "{}.png".format(imagename)),image)
# =============================================================================
# Test Details 
# =============================================================================
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
addition_mode='Sum'    #'Sum'&'Average'&'l1_norm'

model_name="DFRNet    "
# =============================================================================
# Test
# =============================================================================
# for i in range(int(Test_Image_Number/2)):
# for dataset_name in ["TNO512","MSRS","M3FD", "RoadScene512"]: #"TNO","MSRS","M3FD"  "MRI_CT", "MRI_PET", "MRI_SPECT"
for dataset_name in ["TNO512"]: #"TNO","MSRS","M3FD"  "MRI_CT", "MRI_PET", "MRI_SPECT"
# for dataset_name in ["space"]:


    print("\n"*2+"="*80)
    model_name="DFRNet  "
    print("The test result of "+dataset_name+' :')
    test_folder=os.path.join('test_img',dataset_name) 
    test_out_folder=os.path.join('test_result',dataset_name)
    if not os.path.exists(test_out_folder):
             os.makedirs(test_out_folder)

# for img_name in os.listdir(os.path.join(test_data_path, "ir")):
    for img_name in os.listdir(os.path.join(test_folder, "ir")):
        data_IR = image_read_cv2(os.path.join(test_folder, "ir", img_name ), 'GRAY')
        data_VIS = image_read_cv2(os.path.join(test_folder, "vi", img_name ), 'GRAY')
    # for img_name in os.listdir(os.path.join(test_folder,dataset_name.split('_')[0])):
    #     data_IR=image_read_cv2(os.path.join(test_folder,dataset_name.split('_')[1],img_name),mode='GRAY') #[np.newaxis,np.newaxis, ...]/255.0
    #     data_VIS = image_read_cv2(os.path.join(test_folder,dataset_name.split('_')[0],img_name), mode='GRAY') #[np.newaxis,np.newaxis, ...]/255.0
        Fusion_image, _=Test_fusion(data_IR,data_VIS)
        data_Fuse=(Fusion_image-torch.min(Fusion_image))/(torch.max(Fusion_image)-torch.min(Fusion_image)) 
        fi = np.squeeze((data_Fuse * 255).cpu().numpy())       
        # imsave(os.path.join(test_out_folder, "{}".format(img_name)), fi)
        img_save(fi, img_name.split(sep='.')[0], test_out_folder)

        # data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D)
        # data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
        # fi = np.squeeze((data_Fuse * 255).cpu().numpy())  
        # img_save(fi, img_name.split(sep='.')[0], test_out_folder)
        # img_save(fi, img_name.split(sep='.')[0], test_out_folder)              data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
    eval_folder=test_out_folder  
    ori_img_folder=test_folder

    metric_result = np.zeros((12))
#################################################################### 红外可见光图像
    for img_name in os.listdir(os.path.join(ori_img_folder,"ir")):
            ir = image_read_cv2(os.path.join(ori_img_folder,"ir", img_name), 'GRAY')
            vi = image_read_cv2(os.path.join(ori_img_folder,"vi", img_name), 'GRAY')
            fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')
            metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                        , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                        , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                        , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)
                                        , Evaluator.AG(fi), Evaluator.CC(fi, vi, ir)
                                        , Evaluator.MSE(fi, vi, ir), Evaluator.PSNR(fi, vi, ir)])

    metric_result /= len(os.listdir(eval_folder))
    print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM\tAG\tCC\tMSE\tPSNR")
    print(model_name+'\t'+str(np.round(metric_result[0], 2))+'\t'
            +str(np.round(metric_result[1], 2))+'\t'
            +str(np.round(metric_result[2], 2))+'\t'
            +str(np.round(metric_result[3], 2))+'\t'
            +str(np.round(metric_result[4], 2))+'\t'
            +str(np.round(metric_result[5], 2))+'\t'
            +str(np.round(metric_result[6], 2))+'\t'
            +str(np.round(metric_result[7], 2))+'\t'
            +str(np.round(metric_result[8], 2))+'\t'
            +str(np.round(metric_result[9], 2))+'\t'
            +str(np.round(metric_result[10], 2))+'\t'
            +str(np.round(metric_result[11], 2))
            )
    print("="*80)

###################################################################################  医学图像
    # for img_name in os.listdir(os.path.join(ori_img_folder,dataset_name.split('_')[0])):
    #     ir = image_read_cv2(os.path.join(ori_img_folder,dataset_name.split('_')[1], img_name), 'GRAY')
    #     vi = image_read_cv2(os.path.join(ori_img_folder,dataset_name.split('_')[0], img_name), 'GRAY')
    #     fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')
    #     metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
    #                                 , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
    #                                 , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
    #                                 , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)
    #                                 , Evaluator.AG(fi, ir, vi), Evaluator.CC(fi, vi, ir)
    #                                 , Evaluator.Nabf(fi, ir, vi), Evaluator.MSE(fi, vi, ir)
    #                                 , Evaluator.PSNR(fi, vi, ir),
    #                                 ])

    # metric_result /= len(os.listdir(eval_folder))
    # print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
    # print(model_name+'\t'+str(np.round(metric_result[0], 2))+'\t'
    #         +str(np.round(metric_result[1], 2))+'\t'
    #         +str(np.round(metric_result[2], 2))+'\t'
    #         +str(np.round(metric_result[3], 2))+'\t'
    #         +str(np.round(metric_result[4], 2))+'\t'
    #         +str(np.round(metric_result[5], 2))+'\t'
    #         +str(np.round(metric_result[6], 2))+'\t'
    #         +str(np.round(metric_result[7], 2))+'\t'
    #         +str(np.round(metric_result[8], 2))+'\t'
    #         +str(np.round(metric_result[9], 2))+'\t'
    #         +str(np.round(metric_result[10], 2))+'\t'
    #         +str(np.round(metric_result[11], 2))+'\t'
    #         +str(np.round(metric_result[12], 2))+'\t'
    #         +str(np.round(metric_result[13], 2))
    #         )
    # print("="*80)
