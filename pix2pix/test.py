import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from model import Generator,Discriminator
from dataset import Dataset
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as transforms
test_dir = r'/yy/data/depth_map/up'
save_dir = r'/yy/code/pix2pix/output/depth_map/11_15/test'
checkpoint_dir = r'/yy/code/pix2pix/output/checkpoint/11_15_self_atten'
G = Generator()
G.cpu()
G.load_state_dict(torch.load(os.path.join(checkpoint_dir,'best_generator.pth'),map_location='cpu'))
filenames = os.listdir(os.path.join(test_dir,'6_test'))

for name in filenames:
    oppose = Image.open(os.path.join(test_dir,'1_oppose/'+name)).convert('L')
    subject = Image.open(os.path.join(test_dir,'6_test/'+name)).convert('L')
    oppose = oppose.rotate(270)
    subject = subject.rotate(270)
    oppose = transforms.ToTensor()(oppose)
    subject = transforms.ToTensor()(subject)
    inp = torch.cat([oppose,subject],0)
    inp = inp.unsqueeze(0)
    predict = G(inp).squeeze()
    gene_image = predict.detach().numpy()
    gene_image = gene_image*255
    gene_image[gene_image<0] = 0

    gene_image[gene_image>255] = 255
    image = Image.fromarray(gene_image.astype(np.uint8),'L')
    image.save(os.path.join(save_dir,name))




