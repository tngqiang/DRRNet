Codes for ***DFRNet: Unified Dual-Frequency Representation for Enhanced Multimodal Image Fusion***
The visual Computer
[Qiang Tang](https://github.com/tngqiang/).


## Abstract

Multimodal image fusion (MMIF) integrates complementary information from different imaging modalities, such as infrared and visible images, to provide a more discriminative visual representation. However, achieving a balance between global structural consistency and local detail fidelity remains challenging. We propose DFRNet, a unified dual-frequency representation network that models long and short receptive fields collaboratively. Our approach introduces the Dual-Perspective Feature Aggregation Block (DFAB) to extract cross-modal shallow features from both perspectives. Feature maps are decomposed into high and low-frequency branches, with Low-frequency Unet (LUNet) using a long-range Transformer for global dependencies and High-frequency Unet (HUNet) leveraging CNNs for detail enhancement. DFAB fuses these features to generate images that balance structural integrity and fine detail. Experiments on infrared-visible and medical image fusion tasks demonstrate significant improvements in fused image quality, validating DFRNet's effectiveness and practical advantages.

## 🌐 Usage

### ⚙ Network Architecture

Our DFRNet is implemented in ``Fuse.py``.

### 🏊 Training
**1. Virtual Environment**
```
# create virtual environment
conda create -n cddfuse python=3.8.10
conda activate cddfuse
# select pytorch version yourself
# install cddfuse requirements
pip install -r requirements.txt
```

**2. Data Preparation**

Download the MSRS dataset from [this link](https://github.com/Linfeng-Tang/MSRS) and place it in the folder ``'./MSRS_train/'``.

**4. DFRNet Training**
## Training
A pretrained model is available in ```'./Models/Encoder_weight.pkl'``` and ```'./Models/Decoder_weight.pkl'```. We train it on FLIR (180 image pairs) in ```'./Datasets/Train_data```. In the training phase, all images are resize to 128x128 and are transformed to gray pictures.

If you want to re-train this net, you should run ```'train.py'```.

### Testing
The test images used in the paper have been stored in ```'./Test_result/Decoder_weight.pkl'```, ```'./Test_result/Decoder_weight.pkl'``` and ```'./Test_result/FDecoder_weight.pkl'```, respectively.

For other test images, run ```'test.py'``` and find the results in ```'./Test_result/'```.
