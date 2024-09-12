import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from mutil_view_model import *
from dataset_multiview import Dataset
from torch.autograd import Variable
from utils import PerceptualLoss
from Gronet import Generator as Grogenerator
from utils import *
from val_dataset import val_Dataset
import torchvision.transforms as transforms
from consistent_loss import *

batch_size = 8
train_root = r'/yy/data/depth_map/multi_view'
output_dir = r'/yy/code/pix2pix/output/checkpoint/pretrain'
epochs = 100
train_dataset = Dataset('train',train_root)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
Mutil_Generator = Mutil_view_PGenerator()
Mutil_Generator.weight_init(mean=0.0,std=0.02)

Mutil_Generator.cuda()
Mutil_Generator.train()

Mutil_G_optimizer = optim.Adam(Mutil_Generator.parameters(),lr=0.001,betas=(0.5,0.999))
crition_simility = ContrastiveLoss(batch_size,temperature=0.5)
best_loss = 9999
for epoch in range(epochs):
    simility_losses = []
    for data in train_loader:
        oppose_up = data['up']['oppose']
        inp_up = data['up']['inp']
        gt_up = data['up']['gt']
        rio_up = data['up']['rio']
        mask_add = torch.ones((batch_size, 1, 256, 256), dtype=torch.float32)
        up_mask_image1,up_mask1 = add_mask(inp_up, rio_up)
        up_mask_image2,up_mask2 = add_mask(inp_up, rio_up)
        oppose_up = np.array(oppose_up)
        up_mask_image1 = np.array(up_mask_image1)
        up_mask1 = np.array(up_mask1)
        
        up_mask_image2 = np.array(up_mask_image2)
        up_mask2 = np.array(up_mask2)
        
        gt_up = np.array(gt_up)
        oppose_up = transforms.ToTensor()(oppose_up).permute(1, 0, 2).unsqueeze(1)
        up_mask_image1 = transforms.ToTensor()(up_mask_image1).permute(1, 0, 2).unsqueeze(1)
        up_mask1 = transforms.ToTensor()(up_mask1).permute(1, 0, 2).unsqueeze(1)
        up_mask1 = torch.cat((mask_add, up_mask1), axis=1)
        up_mask_image2 = transforms.ToTensor()(up_mask_image2).permute(1, 0, 2).unsqueeze(1)
        up_mask2 = transforms.ToTensor()(up_mask2).permute(1, 0, 2).unsqueeze(1)
        up_mask2 = torch.cat((mask_add, up_mask2), axis=1)
        gt_up = transforms.ToTensor()(gt_up).permute(1, 0, 2).unsqueeze(1)

        oppose_left = data['left']['oppose']
        inp_left = data['left']['inp']
        gt_left = data['left']['gt']
        rio_left = data['left']['rio']
        left_mask_image1,left_mask1 = add_mask(inp_left, rio_left)
        left_mask_image2,left_mask2 = add_mask(inp_left, rio_left)
        oppose_left = np.array(oppose_left)
        left_mask_image1 = np.array(left_mask_image1)
        left_mask1 = np.array(left_mask1)
        
        left_mask_image2 = np.array(left_mask_image2)
        left_mask2 = np.array(left_mask2)
        
        gt_left = np.array(gt_left)
        oppose_left = transforms.ToTensor()(oppose_left).permute(1, 0, 2).unsqueeze(1)
        left_mask_image1 = transforms.ToTensor()(left_mask_image1).permute(1, 0, 2).unsqueeze(1)
        left_mask1 = transforms.ToTensor()(left_mask1).permute(1,0,2).unsqueeze(1)
        left_mask1 = torch.cat((mask_add, left_mask1), axis=1)
        left_mask_image2 = transforms.ToTensor()(left_mask_image2).permute(1, 0, 2).unsqueeze(1)
        left_mask2 = transforms.ToTensor()(left_mask2).permute(1, 0, 2).unsqueeze(1)
        left_mask2 = torch.cat((mask_add, left_mask2), axis=1)
        gt_left = transforms.ToTensor()(gt_left).permute(1, 0, 2).unsqueeze(1)

        oppose_right = data['right']['oppose']
        inp_right = data['right']['inp']
        gt_right = data['right']['gt']
        rio_right = data['right']['rio']
        right_mask_image1,right_mask1 = add_mask(inp_right, rio_right)
        right_mask_image2,right_mask2 = add_mask(inp_right, rio_right)
        oppose_right = np.array(oppose_right)
        right_mask_image1 = np.array(right_mask_image1)
        right_mask1 = np.array(right_mask1)
        
        right_mask_image2 = np.array(right_mask_image2)
        right_mask2 = np.array(right_mask2)
        
        gt_right = np.array(gt_right)
        oppose_right = transforms.ToTensor()(oppose_right).permute(1, 0, 2).unsqueeze(1)
        right_mask_image1 = transforms.ToTensor()(right_mask_image1).permute(1, 0, 2).unsqueeze(1)
        right_mask1 = transforms.ToTensor()(right_mask1).permute(1,0,2).unsqueeze(1)
        right_mask1 = torch.cat((mask_add, right_mask1), axis=1)
        right_mask_image2 = transforms.ToTensor()(right_mask_image2).permute(1, 0, 2).unsqueeze(1)
        right_mask2 = transforms.ToTensor()(right_mask2).permute(1, 0, 2).unsqueeze(1)
        right_mask2 = torch.cat((mask_add, right_mask2), axis=1)
        gt_right = transforms.ToTensor()(gt_right).permute(1, 0, 2).unsqueeze(1)

        oppose_up, up_mask_image1,gt_up = Variable(oppose_up.cuda()), Variable(up_mask_image1.cuda()),Variable(gt_up.cuda())
        oppose_left, left_mask_image1,gt_left = Variable(oppose_left.cuda()), Variable(left_mask_image1.cuda()), Variable(gt_left.cuda())
        oppose_right, right_mask_image1, gt_right = Variable(oppose_right.cuda()), Variable(right_mask_image1.cuda()), Variable(gt_right.cuda())
        up_mask1,up_mask2 = Variable(up_mask1.cuda()),Variable(up_mask2.cuda())
        left_mask1, left_mask2 = Variable(left_mask1.cuda()), Variable(left_mask2.cuda())
        right_mask1, right_mask2 = Variable(right_mask1.cuda()), Variable(right_mask2.cuda())
        up_mask_image2 = Variable(up_mask_image2.cuda())
        left_mask_image2 = Variable(left_mask_image2.cuda())
        right_mask_image2 =  Variable(right_mask_image2.cuda())

        up1 = torch.cat([oppose_up,up_mask_image1],1)
        left1 = torch.cat([oppose_left,left_mask_image1],1)
        right1 = torch.cat([oppose_right,right_mask_image1],1)

        up2 = torch.cat([oppose_up, up_mask_image2], 1)
        left2 = torch.cat([oppose_left, left_mask_image2], 1)
        right2 = torch.cat([oppose_right, right_mask_image2], 1)

        #train_encoder
        Mutil_G_optimizer.zero_grad()
        up_map1,left_map1,right_map1 = Mutil_Generator(up1,left1,right1,up_mask1,left_mask1,right_mask1)
        up_map2, left_map2, right_map2 = Mutil_Generator(up2, left2, right2,up_mask2,left_mask2,right_mask2)
        up_map1,left_map1,right_map1 = up_map1.squeeze(),left_map1.squeeze(),right_map1.squeeze()
        up_map2, left_map2, right_map2 = up_map2.squeeze(), left_map2.squeeze(), right_map2.squeeze()

        loss_up = crition_simility(up_map1,up_map2)
        loss_left = crition_simility(left_map1,left_map2)
        loss_right = crition_simility(right_map1,right_map2)
        loss = loss_up+loss_left+loss_right

        loss.backward()
        Mutil_G_optimizer.step()
        simility_losses.append(loss.item())
    loss_mean = sum(simility_losses)/len(simility_losses)
    print("epoch-{}:,loss:{:.4}".format(epoch+1,loss_mean))
    if (loss_mean<best_loss):
        torch.save(Mutil_Generator.state_dict(), os.path.join(output_dir, 'pretrain_best_model.pth'))
torch.save(Mutil_Generator.state_dict(),os.path.join(output_dir,'pretrain_model.pth'))








