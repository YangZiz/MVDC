import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from model import Pix2pix_Generator,Pix2pix_Discriminator
from pix2pix_dataset import Dataset
from torch.autograd import Variable
from utils import PerceptualLoss
from utils import *

batch_size = 16
ROOT = r'/yy/data/depth_map/multi_view/up'
output_dir = r'/yy/code/pix2pix/output/checkpoint/3_2_pix2pix'
epochs = 200
L1_lambda = 100
Lpg_lambda = 50
save_interv = 50
layer_index = [3,8,15,22]

train_dataset = Dataset('train',ROOT)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

D = Pix2pix_Discriminator(in_channels=2)
G = Pix2pix_Generator()
D.weight_init(mean=0.0,std=0.02)
G.weight_init(mean=0.0,std=0.02)
criterionL1 = nn.L1Loss().cuda()
criterionMSE = nn.MSELoss().cuda()
criterionBCE = nn.BCELoss().cuda()
loss_func = nn.MSELoss().cuda()
criterionPG = PerceptualLoss(loss_func,layer_index)

G.cuda()
D.cuda()

G.train()
D.train()
G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=0.00001, betas=(0.5, 0.999))


train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
G_best_loss = 999999999999

for epoch in range(epochs):
    print('Epoch [%d/%d]' % (epoch+1,epochs))
    D_losses = []
    G_losses = []

    for data in train_loader:
        D.zero_grad()
        inp = data['inp']
        label = data['gt']
        inp,label = Variable(inp.cuda()),Variable(label.cuda())
        D_real = D(torch.cat([inp,label],1))
        #损失
        D_real_loss = D_real.mean()

        G_result = G(inp)
        D_fake = D(torch.cat([inp,G_result],1))
        D_fake_loss = D_fake.mean()
        gradient_penalty = calculate_gradient_penalty(D, torch.cat([inp, label], 1), torch.cat([inp, G_result], 1))

        D_train_loss = D_fake_loss-D_real_loss+gradient_penalty
        D_train_loss.backward()
        D_optimizer.step()
        train_hist['D_losses'].append(D_train_loss.item())
        D_losses.append(D_train_loss.item())



        #train generator
        G.zero_grad()
        G_result = G(inp)
        D_fake = D(torch.cat([inp,G_result],1))
        D_fake_loss_G = D_fake.mean()

        #生成器损失
        loss_PG = criterionPG(torch.cat([G_result,G_result,G_result],dim = 1),torch.cat([label,label,label],dim = 1))

        G_train_loss = -1 * D_fake_loss_G + L1_lambda*criterionL1(G_result,label)+Lpg_lambda*loss_PG

        G_train_loss.backward()
        G_optimizer.step()

        train_hist['G_losses'].append(G_train_loss.item())

        G_losses.append(G_train_loss.item())
        if G_train_loss<G_best_loss:
            G_best_loss = G_train_loss
            torch.save(G.state_dict(),os.path.join(output_dir,'best_generator.pth'))
            torch.save(D.state_dict(),os.path.join(output_dir,'best_discriminator.pth'))
        if (epoch+1)%save_interv == 0:
            torch.save(G.state_dict(), os.path.join(output_dir, '{}_generator.pth'.format(epoch+1)))
            torch.save(D.state_dict(), os.path.join(output_dir, '{}_discriminator.pth'.format(epoch+1)))

    print("epoch-{};,D_loss : {:.4}, G_loss : {:.4}".format(epoch+1,sum(D_losses)/len(D_losses),sum(G_losses)/len(G_losses)))




