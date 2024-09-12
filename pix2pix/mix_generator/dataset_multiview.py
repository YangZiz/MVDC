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
        self.data['up'] = {}
        self.data['left'] = {}
        self.data['right'] = {}
        self.root = image_dir
        self.up_dir = os.path.join(self.root,'up')
        self.left_dir = os.path.join(self.root,'left')
        self.right_dir = os.path.join(self.root,'right')
        self.filenames = [x for x in os.listdir(os.path.join(self.up_dir,'2_object/train'))]

    def __getitem__(self,index):
        #up
        oppose = Image.open(os.path.join(self.up_dir,'1_oppose',self.filenames[index])).convert('L')
        oppose = oppose.rotate(270)
        inp = Image.open(os.path.join(self.up_dir,'2_object/train',self.filenames[index])).convert('L')
        inp = inp.rotate(270)
        mask = Image.open(os.path.join(self.up_dir,'13_new_mask',self.filenames[index])).convert('L')
        mask = mask.rotate(270)
        mask = np.array(mask)
        self.data['up']['mask'] = mask
        oppose = np.array(oppose)
        inp = np.array(inp)
        self.data['up']['oppose'] = oppose
        self.data['up']['inp'] = inp

        rio = Image.open(os.path.join(self.up_dir,'10_rio', self.filenames[index])).convert('L')
        rio = rio.rotate(270)
        rio = np.array(rio)
        rio[rio != 0] = 1

        self.data['up']['rio'] = rio
        if self.mode == 'train':
            gt = Image.open(os.path.join(self.up_dir,'2_object/train',self.filenames[index])).convert('L')
            gt = gt.rotate(270)
            gt = np.array(gt)
            self.data['up']['gt'] = gt

        #left
        oppose = Image.open(os.path.join(self.left_dir, '1_oppose', self.filenames[index])).convert('L')
        oppose = oppose.rotate(270)
        inp = Image.open(os.path.join(self.left_dir, '2_object/train', self.filenames[index])).convert('L')
        inp = inp.rotate(270)
        mask = Image.open(os.path.join(self.left_dir, '13_new_mask', self.filenames[index])).convert('L')
        mask = mask.rotate(270)
        mask = np.array(mask)
        self.data['left']['mask'] = mask
        oppose = np.array(oppose)
        inp = np.array(inp)
        self.data['left']['oppose'] = oppose
        self.data['left']['inp'] = inp
        rio = Image.open(os.path.join(self.left_dir, '10_rio', self.filenames[index])).convert('L')
        rio = rio.rotate(270)
        rio = np.array(rio)
        rio[rio != 0] = 1
        self.data['left']['rio'] = rio
        if self.mode == 'train':
            gt = Image.open(os.path.join(self.left_dir, '2_object/train', self.filenames[index])).convert('L')
            gt = gt.rotate(270)
            gt = np.array(gt)
            self.data['left']['gt'] = gt

        #right
        oppose = Image.open(os.path.join(self.right_dir, '1_oppose', self.filenames[index])).convert('L')
        oppose = oppose.rotate(270)
        inp = Image.open(os.path.join(self.right_dir, '2_object/train', self.filenames[index])).convert('L')
        inp = inp.rotate(270)
        oppose = np.array(oppose)
        inp = np.array(inp)
        mask = Image.open(os.path.join(self.right_dir, '13_new_mask', self.filenames[index])).convert('L')
        mask = mask.rotate(270)
        mask = np.array(mask)
        self.data['right']['mask'] = mask
        self.data['right']['oppose'] = oppose
        self.data['right']['inp'] = inp
        rio = Image.open(os.path.join(self.right_dir, '10_rio', self.filenames[index])).convert('L')
        rio = rio.rotate(270)
        rio = np.array(rio)
        rio[rio != 0] = 1
        self.data['right']['rio'] = rio
        if self.mode == 'train':
            gt = Image.open(os.path.join(self.right_dir, '2_object/train', self.filenames[index])).convert('L')
            gt = gt.rotate(270)
            gt = np.array(gt)
            self.data['right']['gt'] = gt


        return self.data

    def __len__(self):
        return len(self.filenames)


