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
        self.gt_dir = os.path.join(image_dir,'3_train')
        self.rio_dir = os.path.join(image_dir,'10_rio')
        self.filenames = [x for x in os.listdir(self.gt_dir)]

    def __getitem__(self,index):
        oppose = Image.open(os.path.join(self.oppose_dir,self.filenames[index])).convert('L')
        oppose = np.array(oppose)
        #oppose = transforms.ToTensor()(oppose)
        self.data['oppose'] = oppose
        gt = Image.open(os.path.join(self.gt_dir,self.filenames[index])).convert('L')
        gt = np.array(gt)
        partial = Image.open(os.path.join(self.gt_dir,self.filenames[index])).convert('L')
        partial = np.array(partial)

        self.data['partial'] = partial
        #gt = transforms.ToTensor()(gt)
        self.data['gt'] = gt
        rio = Image.open(os.path.join(self.rio_dir, self.filenames[index])).convert('L')
        rio = np.array(rio)
        rio[rio!=0] = 1

        self.data['rio'] = rio

        return self.data

    def __len__(self):
        return len(self.filenames)


