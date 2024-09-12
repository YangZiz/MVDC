import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from Gronet import Generator,Discriminator
from gro_dataset import Dataset
from torch.autograd import Variable

batch_size = 4
ROOT = r'/yy/data/depth_map/multi_view/up'
output_dir = r'/yy/code/pix2pix/output/checkpoint/gronet'
epochs = 500
L1_lambda = 50

train_dataset = Dataset('train',ROOT)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

D = Discriminator()
G = Generator()
D.weight_init(mean=0.0,std=0.02)
G.weight_init(mean=0.0,std=0.02)
criterionL1 = nn.L1Loss().cuda()
criterionMSE = nn.MSELoss().cuda()
criterionBCE = nn.BCELoss().cuda()
loss_func = nn.MSELoss().cuda()

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
        D_real = D(inp,label).squeeze()
        #损失
        D_real_loss = criterionBCE(D_real,Variable(torch.ones(D_real.size()).cuda()))

        G_result = G(inp)
        D_fake = D(inp,G_result).squeeze()
        D_fake_loss = criterionBCE(D_fake,Variable(torch.zeros(D_fake.size()).cuda()))

        D_train_loss = (D_fake_loss+D_real_loss) * 0.5
        D_train_loss.backward()
        D_optimizer.step()
        train_hist['D_losses'].append(D_train_loss.item())
        D_losses.append(D_train_loss.item())


        #train generator
        G.zero_grad()
        G_result = G(inp)
        D_fake = D(inp,G_result).squeeze()
        #生成器损失
        G_train_loss = criterionBCE(D_fake,Variable(torch.ones(D_fake.size()).cuda())) + L1_lambda*criterionL1(G_result,label)

        G_train_loss.backward()
        G_optimizer.step()

        train_hist['G_losses'].append(G_train_loss.item())

        G_losses.append(G_train_loss.item())
    G_losses_mean = sum(G_losses)/len(G_losses)
    if G_losses_mean<G_best_loss:
        G_best_loss = G_train_loss
        torch.save(G.state_dict(),os.path.join(output_dir,'best_generator.pth'))
        torch.save(D.state_dict(),os.path.join(output_dir,'best_discriminator.pth'))

    print("epoch-{};,D_loss : {:.4}, G_loss : {:.4}".format(epoch+1,sum(D_losses)/len(D_losses),sum(G_losses)/len(G_losses)))




