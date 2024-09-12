import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from Gronet import Generator,Discriminator
from dataset import Dataset
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as transforms
test_dir = r'/home/jianjian/yangxunyu/yzz2/pix2pix/output/test/2_object'
save_dir = r'/home/jianjian/yangxunyu/yzz2/pix2pix/output/test/8_occlusal'
checkpoint_dir = r'/home/jianjian/yangxunyu/yzz2/pix2pix/output/gronet'
G = Generator()
G.cpu()
G.load_state_dict(torch.load(os.path.join(checkpoint_dir,'best_generator.pth'),map_location='cpu'))
count = 1
filenames = os.listdir(test_dir)

for name in filenames:
    subject = Image.open(os.path.join(test_dir,name)).convert('L')
    subject = transforms.ToTensor()(subject)
    inp = subject
    inp = inp.unsqueeze(0)
    predict = G(inp).squeeze()
    gene_image = predict.detach().numpy()
    gene_image = gene_image*255
    gene_image[gene_image<0] = 0

    gene_image[gene_image>255] = 255
    image = Image.fromarray(gene_image.astype(np.uint8),'L')
    image.save(os.path.join(save_dir,name))




