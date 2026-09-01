import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from timm.models.layers import trunc_normal_
import os
import numpy as np
import time
from PIL import Image
from dsntnn import *

import matplotlib.pyplot as plt

from eq_module import ASPP
from msreg import MSREG
from mdDecoder import MDDecoder
from sa import SA

def set_seed(seed=42):
    torch.manual_seed(seed)
    
set_seed(42)


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
 
        self.encoder = MSREG(
                        depth=642, 
                        baseWidth=64, 
                        scale=2)
        
        self.sa = SA(2048)
        
        self.aspp = ASPP(in_channels=2048, output_stride=8)
    
        self.head = MDDecoder(in_channels=[256],
                                    in_index=[0],
                                    channels=256,
                                    md_channels=256,
                                    md_kwargs=dict(MD_R=16),
                                    dropout_ratio=0.1,
                                    num_classes=150,
                                    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                                    align_corners=False)
        
        self.decoder_stem = nn.Conv2d(256,20,1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """initialization"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, img, name):
        
        x = self.encoder(img)
        
        x = self.sa(x)
       
        x = self.aspp(x)  # B*256*H*W
        x = x.tensor
        
        x = self.head(x)
        
        x = F.interpolate(x, size=(300, 400), mode='bilinear', align_corners=True)
        
        x = self.decoder_stem(x)
       
        heatmaps = flat_softmax(x)
       
        coords = dsnt(heatmaps)
    
        return coords, heatmaps


class SAWRD_Net(nn.Module):
    def __init__(self,
                 weight):

        super(SAWRD_Net,self).__init__()
        self._net = Model().to(device)
        self._net.load_state_dict(torch.load(weight, map_location=device)['net'])
        for parameter in self._net.parameters():
            parameter.requires_grad = False

    def forward(self,x,name):
        coords, heatmaps = self._net(x,name)
        return coords, heatmaps


def get_file(img_path,label_path):
    img_lst = os.listdir(img_path)
    data_lst = []
    for img in img_lst:
        img_name = img[:-4]
        lab = img_name + '.npy'
        img_file_path = os.path.join(img_path,img)
        label_file_path = os.path.join(label_path,lab)
        data_lst.append((img_file_path,label_file_path,img_name))
    return data_lst

class JointsDataSet(Dataset):
    def __init__(self,data_lst,data_augment=None):
        self.data_lst = data_lst
        self.data_augment = data_augment
        print(f'got {len(data_lst)} images and ground-truths')

    def __getitem__(self,index):
        img = Image.open(self.data_lst[index][0])
        lab = np.load(self.data_lst[index][1])
        name = self.data_lst[index][2]
        img = TF.to_tensor(img).div(255)
        lab = torch.from_numpy(lab)
        image_size = [img.shape[2],img.shape[1]]#[400,300]
        lab_tensor = (lab * 2 + 1) / torch.Tensor(image_size) - 1
        if self.data_augment:
            img,lab_tensor = self.data_augment(img,lab_tensor)
        return img,lab_tensor,name

    def __len__(self):
        return len(self.data_lst)

img_root_path = '../datasets/WRSD/images'
label_root_path = '../datasets/WRSD/annotations'

data_lst = get_file(img_root_path,label_root_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lr = 5e-4
weight_decay = 1e-4
train_bs = 2
num_epochs = 100

train_mse_lst = []
val_mse_lst = []

evaluate_interval = 20
save_interval = 50

weight = []
for j in range(3):
    pweight='./results/model_sawrdnet/'+str(j)+'_model_epoch100.pth'
    weight.append(pweight)

coords_path = './results/test/'
if not os.path.exists(coords_path):
    os.makedirs(coords_path)

best_val_loss = 100.0
time_lst = []


kf = KFold(n_splits=3, random_state=42, shuffle=True)
for j, (train_index, val_index) in enumerate(kf.split(data_lst)):
    net = SAWRD_Net(weight[j]).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_lst = [data_lst[ii] for ii in train_index]
    val_lst = [data_lst[ii] for ii in val_index]
    train_set = JointsDataSet(train_lst)
    val_set = JointsDataSet(val_lst)
    train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    print(f'{j} flod:evaluate model on validation set......')

    net.eval()
    val_mse = 0
    start = time.time()
    with torch.no_grad():
        for img,target_var,name in val_loader:
            img = img.to(device)
            target_var = target_var.to(device)
            
            coords, heatmaps = net(img,name) 
            imgsize_tensor = torch.Tensor([img.shape[3], img.shape[2]]).to(device)
            save_coords = ((coords+1) * imgsize_tensor - 1)/2
            # Per-location euclidean losses
            euc_losses = euclidean_losses(coords, target_var)
            # Per-location regularization losses
            reg_losses = js_reg_losses(heatmaps, target_var, sigma_t=1.0)
            # Per-location focal losses
            focal_losses = focal_reg_losses(heatmaps, target_var, sigma_t=1.0,gamma = 1)
            # Combine losses into an overall loss
            val_loss = average_loss(euc_losses + reg_losses + focal_losses)
            val_mse += val_loss.item()
            # Save results on test set
            name = ''.join(name)
            np.save(coords_path+f'val_100_{name}.npy',save_coords.cpu().numpy()[0])
    end = time.time()
    time_lst.append(end-start)
    print("inference time:",time_lst)
    mean_val_loss = val_mse/len(val_set)
    val_mse_lst.append(mean_val_loss)
    print(f'{j} flod:loss on val set{mean_val_loss}')