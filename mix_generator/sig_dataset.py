import numpy as np
import random
import os
import torch
import  torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

class Dataset(data.Dataset):
    def __init__(self,mode,image_dir):
        super(Dataset,self).__init__()
        self.mode = mode
        self.data = {}
        self.root = image_dir
        self.oppose_dir = os.path.join(image_dir,'1_oppose')
        self.inp_dir = os.path.join(image_dir,'5_inp')
        if self.mode == 'train':
            self.gt_dir = os.path.join(image_dir,'2_object')
        self.filenames = [x for x in os.listdir(self.inp_dir)]

    def __getitem__(self,index):
        oppose = Image.open(os.path.join(self.oppose_dir,self.filenames[index])).convert('L')
        inp = Image.open(os.path.join(self.inp_dir,self.filenames[index])).convert('L')
        oppose = transforms.ToTensor()(oppose)
        inp = transforms.ToTensor()(inp)
        self.data['inp'] = torch.cat([oppose,inp],0)
        if self.mode == 'train':
            gt = Image.open(os.path.join(self.gt_dir,self.filenames[index])).convert('L')
            gt = transforms.ToTensor()(gt)
            self.data['gt'] = gt
        return self.data

    def __len__(self):
        return len(self.filenames)


