import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from model import Pix2pix_Generator,Pix2pix_Discriminator
from pix2pix_dataset import Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
from utils import PerceptualLoss
from utils import *

ROOT = r'/yy/data/depth_map/multi_view/up'
save_root = r'/yy/code/pix2pix/output/depth_map/3_2_pix2pix_test'
checkpoint_path = r'/yy/code/pix2pix/output/checkpoint/3_2_pix2pix/200_generator.pth'
G = Pix2pix_Generator()
G.load_state_dict(torch.load(checkpoint_path,map_location='cpu'))
G.eval()
file_names = os.listdir(os.path.join(ROOT,'2_object/test'))
for name in file_names:
    image_path = os.path.join(ROOT,'5_inp',name)
    save_path = os.path.join(save_root,name)
    inp = Image.open(image_path).convert('L')
    inp = transforms.ToTensor()(inp)
    inp = inp.unsqueeze(0)
    predict = G(inp).squeeze().detach().numpy()
    predict = predict*255
    predict[predict<0] = 0
    predict[predict>255] = 255
    predict_img = Image.fromarray(predict.astype(np.uint8), 'L')
    predict_img.save(save_path)
