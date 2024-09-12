import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from mutil_view_model import *
from dataset_multiview import Dataset
from torch.autograd import Variable
from utils import PerceptualLoss
from Gronet import Generator as Grogenerator
from utils import *
from val_dataset import val_Dataset
import torchvision.transforms as transforms
from consistent_loss import *
import sig_dataset
ROOT = r''
L1_lambda = 100
Lgro_lambda = 50
Lpg_lambda = 50
save_interv = 20
Lconstra_lambda = 100

train_dataset = Dataset('train',ROOT)
val_root = r'/yy/data/depth_map'
output_dir = r'/yy/code/pix2pix/output/checkpoint/12_10_con'
gro_checkpoint = r''
#gro_checkpoint = r'/home/jianjian/yangxunyu/yzz2/pix2pix/output/gronet/best_generator.pth'
output_image = r'/yy/code/pix2pix/output/depth_map/12_10_con'
batch_size = 8
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_dataset = val_Dataset('val',val_root)
val_loader = DataLoader(val_dataset,batch_size=8,shuffle=False)
G = wGenerator()
D = WDiscriminator(in_channels=3)
LD = LocalWDiscriminator()
Gro = Grogenerator()
epochs = 500
best_loss = 999999
now_epoch = 0
G.weight_init(mean = 0.0,std= 0.02)
D.weight_init(mean = 0.0,std= 0.02)
LD.weight_init(mean = 0.0,std= 0.02)
loss_func = nn.MSELoss()
criterionPG = PerceptualLoss(loss_func,layer_indexs=[3,8,15,22])
L1_lambda = 100
Lgro_lambda = 50
Lpg_lambda = 50
criterionL1 = nn.L1Loss().cuda()
criterionMSE = nn.MSELoss().cuda()
criterionBCE = nn.BCELoss().cuda()
G.cuda()
D.cuda()
LD.cuda()
G.train()
D.train()
LD.train()
Gro.cuda()
Gro.load_state_dict(torch.load(gro_checkpoint))
Gro.eval()
G_optimizer = optim.Adam(G.parameters(),lr=0.0002,betas=(0.5,0.999))
D_optimizer = optim.Adam(D.parameters(),lr = 0.00006,betas=(0.5,0.999))
LD_optimizer = optim.Adam(LD.parameters(),lr = 0.00006,betas=(0.5,0.999))
for epoch in range(now_epoch,epochs):
    print('Epoch [%d/%d]' % (epoch + 1, epochs))
    G_losses = []
    D_losses = []
    LD_losses = []
    for data in train_loader:
        inp = data['inp']
        gt = data['gt']
        inp_local = inp[:,:,64:192,64:192]
        gt_local = gt[:,:,64:192,64:192]
        #train_D
        D.zero_grad()
        D_G_result = G(inp)
        D_real = D(torch.cat([inp,gt],1))
        D_real_loss = D_real.mean()
        D_fake = D(torch.cat([inp,D_G_result],1))
        D_fake_loss = D_fake.mean()
        D_gradient_penalty = calculate_gradient_penalty(D,torch.cat([inp,gt],1),torch.cat([inp,D_G_result],1))
        D_train_loss = D_fake_loss-D_real_loss+D_gradient_penalty

        D_train_loss.backward()
        D_optimizer.step()
        D_losses.append(D_train_loss.item())

        #train_LD
        LD.zero_grad()
        LD_G_result = G(inp)
        LD_G_result_local = LD_G_result[:,:,64:192,64:192]
        LD_real = LD(torch.cat([inp_local,gt_local],1))
        LD_real_loss = LD_real.mean()
        LD_fake = LD(torch.cat[inp_local,LD_G_result_local],1)
        LD_fake_loss = LD_fake.mean()
        LD_gradient_penalty = calculate_gradient_penalty(LD,torch.cat([inp_local,gt_local],1),torch.cat([inp_local,LD_G_result_local],1))
        LD_train_loss = LD_fake_loss - LD_real_loss + LD_gradient_penalty
        LD_train_loss.backward()
        LD_optimizer.step()
        LD_losses.append(LD_train_loss.item())


        #train_G
        G.zero_grad()
        G_result = G(inp)
        D_fake_G = D(torch.cat([inp,G_result],1))
        G_result_local = G_result[:,:,64:192,64:192]
        LD_fake_G = LD(torch.cat([inp_local,G_result_local],1))
        D_fake_G_loss = D_fake_G.mean()
        LD_fake_G_loss = LD_fake_G.mean()
        gro_fake = Gro(G_result)
        gro_real = Gro(gt)
        loss_PG = criterionPG(torch.cat([G_result,G_result,G_result],dim = 1),torch.cat([gt,gt,gt],dim = 1))
        G_train_loss = -1*D_fake_G_loss-LD_fake_G_loss + L1_lambda*criterionL1(G_result,gt) + Lpg_lambda*loss_PG + Lgro_lambda*criterionL1(gro_fake,gro_real)
        G_train_loss.backward()
        G_optimizer.step()
        G_losses.append(G_train_loss.item())

    D_losses_mean = sum(D_losses)/len(D_losses)
    LD_losses_mean = sum(LD_losses)/len(LD_losses)
    G_losses_mean = sum(G_losses)/len(G_losses)

    now_rmse = validate_sig(val_loader, G)

    if now_rmse < best_loss:
        best_loss = now_rmse
        torch.save({'epoch': epoch,
                    'model': G.state_dict(),
                    'now_rmse': now_rmse
                    }, os.path.join(output_dir, 'best_generator.pth'))

        torch.save(D.state_dict(), os.path.join(output_dir, 'best_globle_discriminator.pth'))
        torch.save(LD.state_dict(), os.path.join(output_dir, 'best_local_discriminator.pth'))

    if epoch % save_interv == 0:
        torch.save({'epoch': epoch,
                    'model': G.state_dict(),
                    # 'model': Mutil_Generator.state_dict(),
                    'now_rmse': now_rmse
                    }, os.path.join(output_dir, '{}_generator.pth'.format(epoch + 1)))
        # torch.save(D.state_dict(), os.path.join(output_dir, '{}_discriminator.pth'.format(epoch+1)))
        save_inter_sig(inp,G_result, gt, output_image, epoch)
        torch.save(D.state_dict(),
                   os.path.join(output_dir, '{}_globle_discriminator.pth'.format(epoch + 1)))
        torch.save(LD.state_dict(),
                   os.path.join(output_dir, '{}_local_discriminator.pth'.format(epoch + 1)))








