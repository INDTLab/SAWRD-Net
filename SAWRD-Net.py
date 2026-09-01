from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import os
import numpy as np
import time
from PIL import Image
from timm.models.layers import trunc_normal_
from dsntnn import average_loss, dsnt, euclidean_losses, flat_softmax, focal_reg_losses, js_reg_losses

from PIL import Image

from eq_module import ASPP
from msreg import MSREG
from mdDecoder import MDDecoder
from sa import SA

import random
def set_seed(seed=42): #42 45 1024 616 425 318 508 123 6 100
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

    def forward(self,img):

        x = self.encoder(img)
        
        x = self.sa(x)
        
        x = self.aspp(x)
        x = x.tensor
        
        x = self.head(x)
        
        x = F.interpolate(x, size=(300, 400), mode='bilinear', align_corners=True)
        
        x = self.decoder_stem(x)
       
        heatmaps = flat_softmax(x)
        
        coords = dsnt(heatmaps)
    
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 5e-4
weight_decay = 1e-4
train_bs = 2
num_epochs = 100

train_mse_lst = []
val_mse_lst = []

evaluate_interval = 20
save_interval = 50
best_val_loss = 100.0

# Save path for models
snapshot_path = './results/model_sawrdnet/'
# Save path for detection results
heatmap_path = './results/coordsresult_sawrdnet/'
# Save path for loss arrays
log_path = './results/loss_sawrdnet/'
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)
if not os.path.exists(heatmap_path):
    os.makedirs(heatmap_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)

def seed_worker(worker_id):
    worker_seed = 42
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)


kf = KFold(n_splits=3, random_state=6, shuffle=True)
for j, (train_index, val_index) in enumerate(kf.split(data_lst)):
    net = Model().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_lst = [data_lst[ii] for ii in train_index]
    val_lst = [data_lst[ii] for ii in val_index]
    train_set = JointsDataSet(train_lst)
    val_set = JointsDataSet(val_lst)
    train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    best_val_loss = 100.0
    
    for epoch in range(num_epochs):
        train_mse = 0
        tic = time.time()
        net.train()
        for img,target_var,name in train_loader:
            optimizer.zero_grad()
            img = img.to(device)
            target_var = target_var.to(device)
            coords, heatmaps = net(img)
            # Per-location euclidean losses
            euc_losses = euclidean_losses(coords, target_var)
            # Per-location regularization losses
            reg_losses = js_reg_losses(heatmaps, target_var, sigma_t=1.0)
            # Per-location focal losses
            focal_losses = focal_reg_losses(heatmaps, target_var, sigma_t=1.0,gamma = 1)
            # Combine losses into an overall loss
            loss = average_loss(euc_losses + reg_losses + focal_losses)
            loss.backward()
            optimizer.step()
            train_mse += loss.item()*len(img)
        toc = time.time()
        mean_loss = train_mse/len(train_set)
        train_mse_lst.append(mean_loss)
        np.save(log_path + f'{j}_train_mse.npy',np.array(train_mse_lst))

        print(f'{j} flod:[{epoch+1}]/[{num_epochs}] train loss:{mean_loss} time consumption:{toc-tic}s')
        if (epoch+1) % evaluate_interval == 0:
            print(f'{j} flod:evaluate model on validation set......')
            net.eval()
            val_mse = 0
            with torch.no_grad():
                for img,target_var,name in val_loader:
                    img = img.to(device)
                    target_var = target_var.to(device)
                    coords, heatmaps = net(img)
                    imgsize_tensor = torch.Tensor([img.shape[3],img.shape[2]]).to(device)#[400,300]
                    save_coords = ((coords + 1) * imgsize_tensor - 1) / 2
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
                    if (epoch+1) % num_epochs == 0:
                        np.save(heatmap_path+f'val_{epoch+1}_{name}.npy',save_coords.cpu().numpy()[0])
            mean_val_loss = val_mse/len(val_set)
            val_mse_lst.append(mean_val_loss)
            np.save(log_path + f'{j}_val_mse.npy',np.array(val_mse_lst))

            
            print(f'{j} flod:loss on val set{mean_val_loss}')
            # Save best model on test set
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                state_dict = {'net':net.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
                torch.save(state_dict, snapshot_path + f'{j}_best_model.pth')
                print(f'{j} flod:save best val model at epoch{epoch+1}')

        if (epoch+1) % save_interval == 0:
            # Save the model every save_interval
            state_dict = {'net':net.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
            torch.save(state_dict, snapshot_path + f'{j}_model_epoch{epoch+1}.pth')