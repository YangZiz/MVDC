import os.path

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from mutil_view_model import Mutil_view_Generator,Generator
from dataset_multiview import Dataset
from torch.autograd import Variable
#from utils import PerceptualLoss
from PIL import Image
import torchvision.transforms as transforms
#from Gronet import Generator as Grogenerator

test_dir = r'/yy/data/depth_map'
save_dir = r'/yy/data/depth_map/11_22test'
checkpoint_dir = r'/yy/code/pix2pix/output/checkpoint/11_20_multi_view'

# Mutil_Generator = Mutil_view_Generator()
# Mutil_Generator.cpu()
# Mutil_Generator.load_state_dict(torch.load(os.path.join(checkpoint_dir,'best_generator.pth'),map_location='cpu')['model'])

up_G = Generator()
left_G = Generator()
right_G = Generator()
up_G.load_state_dict(torch.load(os.path.join(checkpoint_dir,'best_generator.pth'),map_location='cpu')['up_model'])
left_G.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_generator.pth'),map_location='cpu')['left_model'])
right_G.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_generator.pth'),map_location='cpu')['right_model'])


filenames = os.listdir(os.path.join(test_dir,'up','6_test'))

for name in filenames:
    #up
    save_up_path = os.path.join(save_dir,'up',name)
    save_left_path = os.path.join(save_dir,'left',name)
    save_right_path = os.path.join(save_dir,'right',name)
    up_oppose = Image.open(os.path.join(test_dir,'up','1_oppose',name)).convert('L')
    up_oppose = up_oppose.rotate(270)
    up_inp = Image.open(os.path.join(test_dir,'up','6_test',name)).convert('L')
    up_inp = up_inp.rotate(270)

    #left
    left_oppose = Image.open(os.path.join(test_dir,'left','1_oppose',name)).convert('L')
    left_inp = Image.open(os.path.join(test_dir,'left','5_inp',name)).convert('L')
    left_oppose = left_oppose.rotate(270)
    left_inp = left_inp.rotate(270)

    #right
    right_oppose = Image.open(os.path.join(test_dir,'right','1_oppose',name)).convert('L')
    right_inp = Image.open(os.path.join(test_dir,'right','5_inp',name)).convert('L')
    right_oppose = right_oppose.rotate(270)
    right_inp = right_inp.rotate(270)

    up_oppose = transforms.ToTensor()(up_oppose)
    up_inp = transforms.ToTensor()(up_inp)
    left_oppose = transforms.ToTensor()(left_oppose)
    left_inp = transforms.ToTensor()(left_inp)
    right_oppose = transforms.ToTensor()(right_oppose)
    right_inp = transforms.ToTensor()(right_inp)

    up_oppose = up_oppose.unsqueeze(0).cpu()
    up_inp = up_inp.unsqueeze(0).cpu()
    left_oppose = left_oppose.unsqueeze(0).cpu()
    left_inp = left_inp.unsqueeze(0).cpu()
    right_oppose = right_oppose.unsqueeze(0).cpu()
    right_inp = right_inp.unsqueeze(0).cpu()

    up = torch.cat([up_oppose, up_inp], 1)
    left = torch.cat([left_oppose, left_inp], 1)
    right = torch.cat([right_oppose, right_inp], 1)

    #up_G_result,left_G_result,right_G_result = Mutil_Generator(up,left,right)
    up_G_result = up_G(up)
    left_G_result = left_G(left)
    right_G_result = right_G(right)


    up_G_result = up_G_result.squeeze()
    left_G_result = left_G_result.squeeze()
    right_G_result = right_G_result.squeeze()

    gene_up = up_G_result.detach().numpy()
    gene_left = left_G_result.detach().numpy()
    gene_right = right_G_result.detach().numpy()

    gene_up = gene_up*255
    gene_left = gene_left*255
    gene_right = gene_right*255

    gene_up[gene_up<0] = 0
    gene_left[gene_left<0] = 0
    gene_right[gene_right<0] = 0

    gene_up[gene_up>255] = 255
    gene_left[gene_left>255] = 255
    gene_right[gene_right>255] = 255

    up_image = Image.fromarray(gene_up.astype(np.uint8),'L')
    left_image = Image.fromarray(gene_left.astype(np.uint8),'L')
    right_image = Image.fromarray(gene_right.astype(np.uint8), 'L')

    up_image.save(save_up_path)
    left_image.save(save_left_path)
    right_image.save(save_right_path)







