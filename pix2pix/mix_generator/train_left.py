import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from model import Generator,Discriminator,LocalDiscriminator
from dataset_left import Dataset
from torch.autograd import Variable
from utils import PerceptualLoss


batch_size = 4
ROOT = r'/yy/data/depth_map/left'
output_dir = r'/yy/code/pix2pix/output/checkpoint/left/11_2'
#gro_checkpoint = r'/home/jianjian/yangxunyu/yzz2/pix2pix/output/gronet/best_generator.pth'
epochs = 500
L1_lambda = 100
Lpg_lambda = 50
#Lgro_lambda = 50
save_interv = 50
layer_index = [3,8,15,22]

train_dataset = Dataset('train',ROOT)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

D = Discriminator(in_channels=3)
G = Generator()
LD = LocalDiscriminator()
LD.weight_init(mean=0.0,std = 0.02)
D.weight_init(mean=0.0,std=0.02)
G.weight_init(mean=0.0,std=0.02)
criterionL1 = nn.L1Loss().cuda()
criterionMSE = nn.MSELoss().cuda()
criterionBCE = nn.BCELoss().cuda()
loss_func = nn.MSELoss().cuda()
criterionPG = PerceptualLoss(loss_func,layer_index)

G.cuda()
D.cuda()

LD.cuda()

G.train()
D.train()
G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=0.000002, betas=(0.5, 0.999))
LD_optimizer = optim.Adam(LD.parameters(),lr=0.000002,betas=(0.5,0.999))

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
G_best_loss = 999999999999

for epoch in range(epochs):
    print('Epoch [%d/%d]' % (epoch+1,epochs))
    D_losses = []
    G_losses = []
    LD_losses = []
    for data in train_loader:
        D.zero_grad()
        inp = data['inp']
        label = data['gt']
        inp,label = Variable(inp.cuda()),Variable(label.cuda())
        D_real = D(inp,label).squeeze()
        #损失
        D_real_loss = criterionBCE(D_real,Variable(torch.ones(D_real.size()).cuda()))

        G_result = G(inp)
        D_fake = D(inp,G_result).squeeze()
        D_fake_loss = criterionBCE(D_fake,Variable(torch.zeros(D_fake.size()).cuda()))

        D_train_loss = (D_fake_loss+D_real_loss) * 0.5
        D_real_acu = torch.ge(D_real,0.5).float()
        D_fake_acu = torch.le(D_fake,0.5).float()
        D_total_acu = torch.mean(torch.cat((D_real_acu,D_fake_acu),0))
        if D_total_acu <= 0.8:
            D_train_loss.backward()
            D_optimizer.step()
            train_hist['D_losses'].append(D_train_loss.item())
            D_losses.append(D_train_loss.item())

        #local_discriminator
        LD.zero_grad()
        real_local = label[:,:,64:192,64:192]
        fake = G(inp)
        fake_local = fake[:,:,64:192,64:192]
        #print(fake_local.shape)
        LD_real = LD(real_local).squeeze()
        LD_real_loss = criterionBCE(LD_real,Variable(torch.ones(LD_real.size()).cuda()))
        LD_fake = LD(fake_local).squeeze()
        LD_fake_loss = criterionBCE(LD_fake,Variable(torch.zeros(LD_fake.size()).cuda()))
        LD_train_loss = (LD_real_loss+LD_fake_loss) * 0.5
        LD_real_acu = torch.ge(LD_real, 0.5).float()
        LD_fake_acu = torch.le(LD_fake, 0.5).float()
        LD_total_acu = torch.mean(torch.cat((LD_real_acu, LD_fake_acu), 0))
        if D_total_acu <= 0.8:
            LD_train_loss.backward()
            LD_optimizer.step()
            LD_losses.append(LD_train_loss.item())

        #train generator
        G.zero_grad()
        G_result = G(inp)
        D_fake = D(inp,G_result).squeeze()
        fake_local_G = G_result[:,:,64:192,64:192]
        LD_fake = LD(fake_local_G).squeeze()
        # Gro_fake = Gro(G_result)
        # Gro_real = Gro(label)
        #生成器损失
        loss_PG = criterionPG(torch.cat([G_result,G_result,G_result],dim = 1),torch.cat([label,label,label],dim = 1))
        G_train_loss = criterionBCE(D_fake,Variable(torch.ones(D_fake.size()).cuda())) +criterionBCE(LD_fake,Variable(torch.ones(LD_fake.size()).cuda()))+ L1_lambda*criterionL1(G_result,label)+Lpg_lambda*loss_PG#+Lgro_lambda*criterionL1(Gro_fake,Gro_real)

        G_train_loss.backward()
        G_optimizer.step()

        train_hist['G_losses'].append(G_train_loss.item())

        G_losses.append(G_train_loss.item())
        if G_train_loss<G_best_loss:
            G_best_loss = G_train_loss
            torch.save(G.state_dict(),os.path.join(output_dir,'best_left_generator.pth'))
            torch.save(D.state_dict(),os.path.join(output_dir,'best_left_discriminator.pth'))
            torch.save(LD.state_dict(), os.path.join(output_dir, 'best_left_localdiscriminator.pth'))
        if (epoch+1)%save_interv == 0:
            torch.save(G.state_dict(), os.path.join(output_dir, '{}_left_generator.pth'.format(epoch+1)))
            torch.save(D.state_dict(), os.path.join(output_dir, '{}_left_discriminator.pth'.format(epoch+1)))

    print("epoch-{};,D_loss : {:.4}, G_loss : {:.4}".format(epoch+1,sum(D_losses)/len(D_losses),sum(G_losses)/len(G_losses)))




