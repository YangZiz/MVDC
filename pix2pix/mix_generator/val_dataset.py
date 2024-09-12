import numpy as np
import random
import os
import torch
import  torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

class val_Dataset(data.Dataset):
    def __init__(self,mode,image_dir):
        super(val_Dataset,self).__init__()
        self.mode = mode
        self.data = {}
        self.data['up'] = {}
        self.data['left'] = {}
        self.data['right'] = {}
        self.root = image_dir
        self.up_dir = os.path.join(self.root,'up')
        self.left_dir = os.path.join(self.root,'left')
        self.right_dir = os.path.join(self.root,'right')
        self.filenames = [x for x in os.listdir(os.path.join(self.up_dir,'2_object/test'))]

    def __getitem__(self,index):
        #up
        mask_add = torch.ones((1, 256, 256), dtype=torch.float32)
        oppose = Image.open(os.path.join(self.up_dir,'1_oppose',self.filenames[index])).convert('L')
        inp = Image.open(os.path.join(self.up_dir,'5_inp',self.filenames[index])).convert('L')
        oppose = transforms.ToTensor()(oppose)
        inp = transforms.ToTensor()(inp)
        mask = Image.open(os.path.join(self.up_dir,'13_new_mask',self.filenames[index])).convert('L')
        mask = transforms.ToTensor()(mask)
        mask[mask!=0] = 1
        mask = torch.cat((mask_add, mask), axis=0)
        self.data['up']['oppose'] = oppose
        self.data['up']['inp'] = inp
        self.data['up']['mask'] = mask
        if self.mode == 'val':
            gt = Image.open(os.path.join(self.up_dir,'2_object/test',self.filenames[index])).convert('L')
            gt = transforms.ToTensor()(gt)
            self.data['up']['gt'] = gt

        #left
        oppose = Image.open(os.path.join(self.left_dir, '1_oppose', self.filenames[index])).convert('L')
        inp = Image.open(os.path.join(self.left_dir, '5_inp', self.filenames[index])).convert('L')
        oppose = transforms.ToTensor()(oppose)
        inp = transforms.ToTensor()(inp)
        mask = Image.open(os.path.join(self.left_dir, '13_new_mask', self.filenames[index])).convert('L')
        mask = transforms.ToTensor()(mask)
        mask[mask != 0] = 1
        mask = torch.cat((mask_add, mask), axis=0)
        self.data['left']['oppose'] = oppose
        self.data['left']['inp'] = inp
        self.data['left']['mask'] = mask
        if self.mode == 'val':
            gt = Image.open(os.path.join(self.left_dir, '2_object/test', self.filenames[index])).convert('L')
            gt = transforms.ToTensor()(gt)
            self.data['left']['gt'] = gt

        #right
        oppose = Image.open(os.path.join(self.right_dir, '1_oppose', self.filenames[index])).convert('L')
        inp = Image.open(os.path.join(self.right_dir, '5_inp', self.filenames[index])).convert('L')
        oppose = transforms.ToTensor()(oppose)
        inp = transforms.ToTensor()(inp)
        mask = Image.open(os.path.join(self.right_dir, '13_new_mask', self.filenames[index])).convert('L')
        mask = transforms.ToTensor()(mask)
        mask[mask != 0] = 1

        mask = torch.cat((mask_add,mask),axis=0)
        self.data['right']['oppose'] = oppose
        self.data['right']['inp'] = inp
        self.data['right']['mask'] = mask
        if self.mode == 'val':
            gt = Image.open(os.path.join(self.right_dir, '2_object/test', self.filenames[index])).convert('L')
            gt = transforms.ToTensor()(gt)
            self.data['right']['gt'] = gt


        return self.data

    def __len__(self):
        return len(self.filenames)


