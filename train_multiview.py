import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from model import Generator,Discriminator,LocalDiscriminator,Mutil_Discriminator,Mutil_Generator
from dataset_multiview import Dataset
from torch.autograd import Variable
#from utils import PerceptualLoss
from Gronet import Generator as Grogenerator

batch_size = 4
ROOT = r'/yy/data/depth_map'
output_dir = r'/yy/code/pix2pix/output/checkpoint/11_9_ins'
gro_checkpoint = r'/home/jianjian/yangxunyu/yzz2/pix2pix/output/gronet/best_generator.pth'
epochs = 500
L1_lambda = 100
Lgro_lambda = 50
save_interv = 50
train_dataset = Dataset('train',ROOT)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

up_G = Mutil_Generator()
left_G = Mutil_Generator()
right_G = Mutil_Generator()
up_D = Discriminator(in_channels=3)
left_D = Discriminator(in_channels=3)
right_D = Discriminator(in_channels=3)
D = Mutil_Discriminator(in_channels=3)
Gro = Grogenerator()

up_G.weight_init(mean=0.0,std=0.02)
left_G.weight_init(mean=0.0,std=0.02)
right_G.weight_init(mean=0.0,std=0.02)
up_D.weight_init(mean=0.0,std=0.02)
left_D.weight_init(mean=0.0,std=0.02)
right_D.weight_init(mean=0.0,std=0.02)
D.weight_init(mean=0.0,std=0.02)

criterionL1 = nn.L1Loss().cuda()
criterionMSE = nn.MSELoss().cuda()
criterionBCE = nn.BCELoss().cuda()

up_G.cuda()
left_G.cuda()
right_G.cuda()
up_D.cuda()
left_D.cuda()
right_D.cuda()
D.cuda()
Gro.cuda()
Gro.load_state_dict(torch.load(gro_checkpoint))

up_G.train()
left_G.train()
right_G.train()
up_D.train()
left_D.train()
right_D.train()
D.train()

up_G_optimizer = optim.Adam(up_G.parameters(),lr=0.0002,betas=(0.5,0.999))
left_G_optimizer = optim.Adam(left_G.parameters(),lr = 0.0002,betas=(0.5,0.999))
right_G_optimizer = optim.Adam(right_G.parameters(),lr = 0.0002,betas=(0.5,0.999))
up_D_optimizer = optim.Adam(up_D.parameters(),lr=0.000006,betas=(0.5,0.999))
left_D_optimizer = optim.Adam(left_D.parameters(),lr = 0.000006,betas=(0.5,0.999))
right_D_optimizer = optim.Adam(right_D.parameters(),lr = 0.000006,betas=(0.5,0.999))
D_optimizer = optim.Adam(D.parameters(),lr = 0.000006,betas=(0.5,0.999))

train_hist = {}
train_hist['up_G_losses'] = []
train_hist['left_G_losses'] = []
train_hist['right_G_losses'] = []
train_hist['up_D_losses'] = []
train_hist['left_D_losses'] = []
train_hist['right_D_losses'] = []
train_hist['D_losses'] = []
best_loss = 999999
for epoch in range(epochs):
    print('Epoch [%d/%d]' % (epoch + 1, epochs))
    D_losses = []
    up_G_losses = []
    left_G_losses = []
    right_G_losses = []
    up_D_losses = []
    left_D_losses = []
    right_D_losses = []
    for data in train_loader:
        D.zero_grad()

        oppose_up = data['up']['oppose']
        inp_up = data['up']['inp']
        gt_up = data['up']['gt']

        oppose_left = data['left']['oppose']
        inp_left = data['left']['inp']
        gt_left = data['left']['gt']

        oppose_right = data['right']['oppose']
        inp_right = data['right']['inp']
        gt_right = data['right']['gt']

        oppose_up,inp_up,gt_up = Variable(oppose_up.cuda()),Variable(inp_up.cuda()),Variable(gt_up.cuda())
        oppose_left, inp_left, gt_left = Variable(oppose_left.cuda()), Variable(inp_left.cuda()), Variable(gt_left.cuda())
        oppose_right, inp_right, gt_right = Variable(oppose_right.cuda()), Variable(inp_right.cuda()), Variable(gt_right.cuda())

        #train D
        D_real = D(gt_up,gt_left,gt_right).squeeze()
        D_real_loss = criterionBCE(D_real, Variable(torch.ones(D_real.size()).cuda()))

        z = torch.rand(batch_size,256,16,16).cuda()
        up_e4,_ = up_G(torch.cat([oppose_up,inp_up],1),z)
        left_e4,_ = left_G(torch.cat([oppose_left,inp_left],1),z)
        right_e4,_ = right_G(torch.cat([oppose_right,inp_right],1),z)


        #图卷积换掉
        z = up_e4+left_e4+right_e4

        _,up_G_result = up_G(torch.cat([oppose_up, inp_up], 1), z)
        _, left_G_result = left_G(torch.cat([oppose_left, inp_left], 1), z)
        _,right_G_result = right_G(torch.cat([oppose_right, inp_right], 1), z)

        D_fake = D(up_G_result,left_G_result,right_G_result).squeeze()
        D_fake_loss = criterionBCE(D_fake, Variable(torch.zeros(D_fake.size()).cuda()))

        D_train_loss = (D_fake_loss + D_real_loss) * 0.5
        D_train_loss.backward()
        D_optimizer.step()
        train_hist['D_losses'].append(D_train_loss.item())
        D_losses.append(D_train_loss.item())

        #train up_D
        up_D.zero_grad()
        up_D_real = up_D(torch.cat([oppose_up,inp_up],1),gt_up).squeeze()
        up_D_real_loss = criterionBCE(up_D_real, Variable(torch.ones(up_D_real.size()).cuda()))

        z = torch.rand(batch_size, 256, 16, 16).cuda()
        up_e4, _ = up_G(torch.cat([oppose_up, inp_up], 1), z)
        left_e4, _ = left_G(torch.cat([oppose_left, inp_left], 1), z)
        right_e4, _ = right_G(torch.cat([oppose_right, inp_right], 1), z)

        z = up_e4+left_e4+right_e4

        _, up_G_result = up_G(torch.cat([oppose_up, inp_up], 1), z)
        up_D_fake = up_D(torch.cat([oppose_up,inp_up],1),up_G_result).squeeze()
        up_D_fake_loss = criterionBCE(up_D_fake,Variable(torch.zeros(up_D_fake.size()).cuda()))

        up_D_train_loss = (up_D_fake_loss + up_D_real_loss) * 0.5
        up_D_train_loss.backward()
        up_D_optimizer.step()
        train_hist['up_D_losses'].append(up_D_train_loss.item())
        up_D_losses.append(up_D_train_loss.item())

        # train left_D
        left_D.zero_grad()
        left_D_real = left_D(torch.cat([oppose_left, inp_left], 1), gt_left).squeeze()
        left_D_real_loss = criterionBCE(left_D_real, Variable(torch.ones(left_D_real.size()).cuda()))

        z = torch.rand(batch_size, 256, 16, 16).cuda()
        up_e4, _ = up_G(torch.cat([oppose_up, inp_up], 1), z)
        left_e4, _ = left_G(torch.cat([oppose_left, inp_left], 1), z)
        right_e4, _ = right_G(torch.cat([oppose_right, inp_right], 1), z)

        z = up_e4+left_e4+right_e4

        _, left_G_result = left_G(torch.cat([oppose_left, inp_left], 1), z)

        left_D_fake = left_D(torch.cat([oppose_left, inp_left], 1), left_G_result).squeeze()
        left_D_fake_loss = criterionBCE(left_D_fake, Variable(torch.zeros(left_D_fake.size()).cuda()))

        left_D_train_loss = (left_D_fake_loss + left_D_real_loss) * 0.5
        left_D_train_loss.backward()
        left_D_optimizer.step()
        train_hist['left_D_losses'].append(left_D_train_loss.item())
        left_D_losses.append(left_D_train_loss.item())

        # train right_D
        right_D.zero_grad()
        right_D_real = right_D(torch.cat([oppose_right, inp_right], 1), gt_right).squeeze()
        right_D_real_loss = criterionBCE(right_D_real, Variable(torch.ones(right_D_real.size()).cuda()))

        z = torch.rand(batch_size, 256, 16, 16).cuda()
        up_e4, _ = up_G(torch.cat([oppose_up, inp_up], 1), z)
        left_e4, _ = left_G(torch.cat([oppose_left, inp_left], 1), z)
        right_e4, _ = right_G(torch.cat([oppose_right, inp_right], 1), z)

        z = up_e4+left_e4+right_e4

        _, right_G_result = right_G(torch.cat([oppose_right, inp_right], 1), z)

        right_D_fake = right_D(torch.cat([oppose_right, inp_right], 1), right_G_result).squeeze()
        right_D_fake_loss = criterionBCE(right_D_fake, Variable(torch.zeros(right_D_fake.size()).cuda()))

        right_D_train_loss = (right_D_fake_loss + right_D_real_loss) * 0.5
        right_D_train_loss.backward()
        right_D_optimizer.step()
        train_hist['right_D_losses'].append(right_D_train_loss.item())
        right_D_losses.append(right_D_train_loss.item())


        #train up
        up_G.zero_grad()
        z = torch.rand(batch_size, 256, 16, 16).cuda()
        up_e4, _ = up_G(torch.cat([oppose_up, inp_up], 1), z)
        left_e4, _ = left_G(torch.cat([oppose_left, inp_left], 1), z)
        right_e4, _ = right_G(torch.cat([oppose_right, inp_right], 1), z)

        z = up_e4+left_e4+right_e4

        _, up_G_result = up_G(torch.cat([oppose_up, inp_up], 1), z)
        _, left_G_result = left_G(torch.cat([oppose_left, inp_left], 1), z)
        _, right_G_result = right_G(torch.cat([oppose_right, inp_right], 1), z)
        D_fake = D(up_G_result,left_G_result,right_G_result).squeeze()
        up_D_fake = up_D(torch.cat([oppose_up,inp_up],1),up_G_result).squeeze()
        Gro_fake = Gro(up_G_result)
        Gro_real = Gro(gt_up)

        up_G_train_loss = criterionBCE(D_fake,Variable(torch.ones(D_fake.size()).cuda())) + criterionBCE(up_D_fake,Variable(torch.ones(up_D_fake.size()).cuda())) + L1_lambda*criterionL1(up_G_result,gt_up) + Lgro_lambda*criterionL1(Gro_fake,Gro_real)
        up_G_train_loss.backward()
        up_G_optimizer.step()
        train_hist['up_G_losses'].append(up_G_train_loss.item())
        up_G_losses.append(up_G_train_loss.item())

        #trian left
        left_G.zero_grad()
        z = torch.rand(batch_size, 256, 16, 16).cuda()
        up_e4, _ = up_G(torch.cat([oppose_up, inp_up], 1), z)
        left_e4, _ = left_G(torch.cat([oppose_left, inp_left], 1), z)
        right_e4, _ = right_G(torch.cat([oppose_right, inp_right], 1), z)

        z = up_e4+left_e4+right_e4

        _, up_G_result = up_G(torch.cat([oppose_up, inp_up], 1), z)
        _, left_G_result = left_G(torch.cat([oppose_left, inp_left], 1), z)
        _, right_G_result = right_G(torch.cat([oppose_right, inp_right], 1), z)
        D_fake = D(up_G_result, left_G_result, right_G_result).squeeze()
        left_D_fake = left_D(torch.cat([oppose_left, inp_left], 1), left_G_result).squeeze()

        left_G_train_loss = criterionBCE(D_fake,Variable(torch.ones(D_fake.size()).cuda())) + criterionBCE(left_D_fake,Variable(torch.ones(left_D_fake.size()).cuda())) + L1_lambda * criterionL1(left_G_result,gt_left)
        left_G_train_loss.backward()
        left_G_optimizer.step()
        train_hist['left_G_losses'].append(left_G_train_loss.item())
        left_G_losses.append(left_G_train_loss.item())


        #train right
        right_G.zero_grad()
        z = torch.rand(batch_size, 256, 16, 16).cuda()
        up_e4, _ = up_G(torch.cat([oppose_up, inp_up], 1), z)
        left_e4, _ = left_G(torch.cat([oppose_left, inp_left], 1), z)
        right_e4, _ = right_G(torch.cat([oppose_right, inp_right], 1), z)

        z = up_e4+left_e4+right_e4

        _, up_G_result = up_G(torch.cat([oppose_up, inp_up], 1), z)
        _, left_G_result = left_G(torch.cat([oppose_left, inp_left], 1), z)
        _, right_G_result = right_G(torch.cat([oppose_right, inp_right], 1), z)
        D_fake = D(up_G_result, left_G_result, right_G_result).squeeze()
        right_D_fake = right_D(torch.cat([oppose_right, inp_right], 1), right_G_result).squeeze()

        right_G_train_loss = criterionBCE(D_fake, Variable(torch.ones(D_fake.size()).cuda())) + criterionBCE(right_D_fake,Variable(torch.ones(right_D_fake.size()).cuda())) + L1_lambda * criterionL1(right_G_result, gt_right)
        right_G_train_loss.backward()
        right_G_optimizer.step()
        train_hist['right_G_losses'].append(right_G_train_loss.item())
        right_G_losses.append(right_G_train_loss.item())

    up_G_losses_mean = sum(up_G_losses)/len(up_G_losses)
    left_G_losses_mean = sum(left_G_losses) / len(left_G_losses)
    right_G_losses_mean = sum(right_G_losses) / len(right_G_losses)

    mean_loss = up_G_losses_mean+0.5*left_G_losses_mean+0.5*right_G_losses_mean


    if mean_loss<best_loss:
        best_loss = mean_loss
        torch.save(up_G.state_dict(), os.path.join(output_dir, 'best_up_generator.pth'))
        torch.save(left_G.state_dict(), os.path.join(output_dir, 'best_left_generator.pth'))
        torch.save(right_G.state_dict(), os.path.join(output_dir, 'best_right_generator.pth'))
        torch.save(D.state_dict(), os.path.join(output_dir, 'best_discriminator.pth'))
        torch.save(up_D.state_dict(), os.path.join(output_dir, 'best_up_discriminator.pth'))
        torch.save(left_D.state_dict(), os.path.join(output_dir, 'best_left_discriminator.pth'))
        torch.save(right_D.state_dict(), os.path.join(output_dir, 'best_right_discriminator.pth'))
    if (epoch + 1) % save_interv == 0:
        torch.save(up_G.state_dict(), os.path.join(output_dir, '{}_up_generator.pth'.format(epoch+1)))
        torch.save(left_G.state_dict(), os.path.join(output_dir, '{}_left_generator.pth'.format(epoch+1)))
        torch.save(right_G.state_dict(), os.path.join(output_dir, '{}_right_generator.pth'.format(epoch+1)))
        torch.save(D.state_dict(), os.path.join(output_dir, '{}_discriminator.pth'.format(epoch+1)))
        torch.save(up_D.state_dict(), os.path.join(output_dir, '{}_up_discriminator.pth'.format(epoch + 1)))
        torch.save(left_D.state_dict(), os.path.join(output_dir, '{}_left_discriminator.pth'.format(epoch + 1)))
        torch.save(right_D.state_dict(), os.path.join(output_dir, '{}_right_discriminator.pth'.format(epoch + 1)))
    print("epoch-{};,D_loss : {:.4}, up_G_loss : {:.4}, left_G_loss : {:.4},right_G_loss : {:.4},up_D_loss : {:.4}, left_D_loss : {:.4},right_D_loss : {:.4}".format(epoch+1,sum(D_losses)/len(D_losses),sum(up_G_losses)/len(up_G_losses),sum(left_G_losses)/len(left_G_losses),sum(right_G_losses)/len(right_G_losses),sum(up_D_losses)/len(up_D_losses),sum(left_D_losses)/len(left_D_losses),sum(right_D_losses)/len(right_D_losses)))





