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

batch_size = 2
ROOT = r'/yy/data/depth_map/multi_view'
val_root = r'/yy/data/depth_map/multi_view'
output_dir = r'/yy/code/pix2pix/output/checkpoint/1_18_gro_w_p'
gro_checkpoint = r'/yy/code/pix2pix/output/checkpoint/gronet/best_generator.pth'
output_image = r'/yy/code/pix2pix/output/depth_map/1_18'
pretrain_path = r'/yy/code/pix2pix/output/checkpoint/pretrain/pretrain_best_model.pth'
epochs = 500
L1_lambda = 100
Lgro_lambda = 50
Lpg_lambda = 50
save_interv = 20
train_dataset = Dataset('train',ROOT)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
val_dataset = val_Dataset('val',val_root)
val_loader = DataLoader(val_dataset,batch_size=8,shuffle=False)


Mutil_Generator = Mutil_view_PGenerator()
# up_G = wGenerator()
# left_G = wGenerator()
# right_G = wGenerator()


up_D = WDiscriminator(in_channels=3)
left_D = WDiscriminator(in_channels=3)
right_D = WDiscriminator(in_channels=3)

up_LD = LocalWDiscriminator()
left_LD = LocalWDiscriminator()
right_LD = LocalWDiscriminator()

#D = Mutil_Discriminator(in_channels=3)
Gro = Grogenerator()
best_loss = 999999
now_epoch = 0
conti = 0
#Mutil_Generator.weight_init(mean=0.0,std = 0.02)
if (os.path.isfile(os.path.join(output_dir,'best_generator.pth'))):
    Mutil_Generator.load_state_dict(torch.load(os.path.join(output_dir,'best_generator.pth'))['model'])
    now_epoch = torch.load(os.path.join(output_dir,'best_generator.pth'))['epoch']
    best_loss = torch.load(os.path.join(output_dir,'best_generator.pth'))['now_rmse']
    #D.load_state_dict(torch.load(os.path.join(output_dir,'best_discriminator.pth')))
    up_D.load_state_dict(torch.load(os.path.join(output_dir, 'best_up_discriminator.pth')))
    left_D.load_state_dict(torch.load(os.path.join(output_dir, 'best_left_discriminator.pth')))
    right_D.load_state_dict(torch.load(os.path.join(output_dir, 'best_right_discriminator.pth')))

    up_LD.load_state_dict(torch.load(os.path.join(output_dir, 'best_up_localdiscriminator.pth')))
    left_LD.load_state_dict(torch.load(os.path.join(output_dir, 'best_left_localdiscriminator.pth')))
    right_LD.load_state_dict(torch.load(os.path.join(output_dir, 'best_right_localdiscriminator.pth')))




# elif conti ==1:
#     Mutil_Generator.load_state_dict(torch.load(os.path.join(output_dir, '201_generator.pth'))['model'])
#     now_epoch = 201
#     best_loss = 99
#     # D.load_state_dict(torch.load(os.path.join(output_dir,'best_discriminator.pth')))
#     up_D.load_state_dict(torch.load(os.path.join(output_dir, '201_up_discriminator.pth')))
#     left_D.load_state_dict(torch.load(os.path.join(output_dir, '201_left_discriminator.pth')))
#     right_D.load_state_dict(torch.load(os.path.join(output_dir, '201_right_discriminator.pth')))
#
#     up_LD.load_state_dict(torch.load(os.path.join(output_dir, '201_up_localdiscriminator.pth')))
#     left_LD.load_state_dict(torch.load(os.path.join(output_dir, '201_left_localdiscriminator.pth')))
#     right_LD.load_state_dict(torch.load(os.path.join(output_dir, '201_right_localdiscriminator.pth')))


else:
    # up_G.weight_init(mean=0.0,std=0.02)
    # left_G.weight_init(mean=0.0,std=0.02)
    # right_G.weight_init(mean=0.0,std=0.02)
    Mutil_Generator.weight_init(mean=0.0,std=0.02)
    #Mutil_Generator.load_state_dict(torch.load(pretrain_path))
    up_D.weight_init(mean=0.0,std=0.02)
    left_D.weight_init(mean=0.0,std=0.02)
    right_D.weight_init(mean=0.0,std=0.02)
    #D.weight_init(mean=0.0,std=0.02)

    up_LD.weight_init(mean=0.0, std=0.02)
    left_LD.weight_init(mean=0.0, std=0.02)
    right_LD.weight_init(mean=0.0, std=0.02)

print(best_loss)
criterionL1 = nn.L1Loss().cuda()
criterionMSE = nn.MSELoss().cuda()
criterionBCE = nn.BCELoss().cuda()

consistent_loss_up = Consistent_loss_up_4()
# consistent_loss_left = Consistent_loss_left()
# consistent_loss_right = Consistent_loss_right()

loss_func = nn.MSELoss()
criterionPG = PerceptualLoss(loss_func,layer_indexs=[3,8,15,22])

Mutil_Generator.cuda()
# up_G.cuda()
# left_G.cuda()
# right_G.cuda()
up_D.cuda()
left_D.cuda()
right_D.cuda()
#D.cuda()
up_LD.cuda()
left_LD.cuda()
right_LD.cuda()
Gro.cuda()
Gro.load_state_dict(torch.load(gro_checkpoint))
Gro.eval()

Mutil_Generator.train()
# up_G.train()
# left_G.train()
# right_G.train()
up_D.train()
left_D.train()
right_D.train()
#D.train()
up_LD.train()
left_LD.train()
right_LD.train()

Mutil_G_optimizer = optim.Adam(Mutil_Generator.parameters(),lr=0.00002,betas=(0.5,0.999))
# up_G_optimizer = optim.Adam(up_G.parameters(),lr=0.0002,betas=(0.5,0.999))
# left_G_optimizer = optim.Adam(left_G.parameters(),lr = 0.0002,betas=(0.5,0.999))
# right_G_optimizer = optim.Adam(right_G.parameters(),lr = 0.0002,betas=(0.5,0.999))
up_D_optimizer = optim.Adam(up_D.parameters(),lr=0.000006,betas=(0.5,0.999))
left_D_optimizer = optim.Adam(left_D.parameters(),lr = 0.000006,betas=(0.5,0.999))
right_D_optimizer = optim.Adam(right_D.parameters(),lr = 0.000006,betas=(0.5,0.999))
#D_optimizer = optim.Adam(D.parameters(),lr = 0.00006,betas=(0.5,0.999))
up_LD_optimizer = optim.Adam(up_LD.parameters(),lr=0.000006,betas=(0.5,0.999))
left_LD_optimizer = optim.Adam(left_LD.parameters(),lr = 0.000006,betas=(0.5,0.999))
right_LD_optimizer = optim.Adam(right_LD.parameters(),lr = 0.000006,betas=(0.5,0.999))

train_hist = {}

train_hist['G_losses'] = []
train_hist['up_D_losses'] = []
train_hist['left_D_losses'] = []
train_hist['right_D_losses'] = []
#train_hist['D_losses'] = []
train_hist['up_LD_losses'] = []
train_hist['left_LD_losses'] = []
train_hist['right_LD_losses'] = []

for epoch in range(now_epoch,epochs):
    print('Epoch [%d/%d]' % (epoch + 1, epochs))
    #D_losses = []

    G_losses = []
    G_w_losses = []
    G_l1_losses = []
    G_adv_losses = []
    G_consis_losses = []
    G_gro_losses = []


    up_D_losses = []
    left_D_losses = []
    right_D_losses = []

    up_LD_losses = []
    left_LD_losses = []
    right_LD_losses = []

    up_D_w = []
    left_D_w = []
    right_D_w = []

    up_LD_w = []
    left_LD_w = []
    right_LD_w = []

    #count = 0

    for data in train_loader:
        #D.zero_grad()

        oppose_up = data['up']['oppose']
        inp_up = data['up']['inp']
        gt_up = data['up']['gt']
        rio_up = data['up']['rio']
        mask_add = torch.ones((batch_size, 1, 256, 256), dtype=torch.float32)
        up_mask_image,up_mask = add_mask(inp_up,rio_up)
        oppose_up = np.array(oppose_up)
        up_mask_image = np.array(up_mask_image)
        up_mask = np.array(up_mask)
        gt_up = np.array(gt_up)
        oppose_up = transforms.ToTensor()(oppose_up).permute(1, 0, 2).unsqueeze(1)
        up_mask_image = transforms.ToTensor()(up_mask_image).permute(1, 0, 2).unsqueeze(1)
        gt_up = transforms.ToTensor()(gt_up).permute(1, 0, 2).unsqueeze(1)
        up_mask = transforms.ToTensor()(up_mask).permute(1, 0, 2).unsqueeze(1)
        up_mask = torch.cat((mask_add, up_mask), axis=1)

        oppose_left = data['left']['oppose']
        inp_left = data['left']['inp']
        gt_left = data['left']['gt']
        rio_left = data['left']['rio']
        left_mask_image,left_mask = add_mask(inp_left, rio_left)
        oppose_left = np.array(oppose_left)
        left_mask_image = np.array(left_mask_image)
        left_mask = np.array(left_mask)
        gt_left = np.array(gt_left)
        oppose_left = transforms.ToTensor()(oppose_left).permute(1, 0, 2).unsqueeze(1)
        left_mask_image = transforms.ToTensor()(left_mask_image).permute(1, 0, 2).unsqueeze(1)
        gt_left = transforms.ToTensor()(gt_left).permute(1, 0, 2).unsqueeze(1)
        left_mask = transforms.ToTensor()(left_mask).permute(1, 0, 2).unsqueeze(1)
        left_mask = torch.cat((mask_add, left_mask), axis=1)

        oppose_right = data['right']['oppose']
        inp_right = data['right']['inp']
        gt_right = data['right']['gt']
        rio_right = data['right']['rio']
        right_mask_image,right_mask = add_mask(inp_right, rio_right)
        oppose_right = np.array(oppose_right)
        right_mask_image = np.array(right_mask_image)
        right_mask = np.array(right_mask)
        gt_right = np.array(gt_right)
        oppose_right = transforms.ToTensor()(oppose_right).permute(1, 0, 2).unsqueeze(1)
        right_mask_image = transforms.ToTensor()(right_mask_image).permute(1, 0, 2).unsqueeze(1)
        gt_right = transforms.ToTensor()(gt_right).permute(1, 0, 2).unsqueeze(1)
        right_mask = transforms.ToTensor()(right_mask).permute(1, 0, 2).unsqueeze(1)
        right_mask = torch.cat((mask_add, right_mask), axis=1)
        oppose_up,up_mask_image,gt_up = Variable(oppose_up.cuda()),Variable(up_mask_image.cuda()),Variable(gt_up.cuda())
        oppose_left, left_mask_image, gt_left = Variable(oppose_left.cuda()), Variable(left_mask_image.cuda()), Variable(gt_left.cuda())
        oppose_right, right_mask_image, gt_right = Variable(oppose_right.cuda()), Variable(right_mask_image.cuda()), Variable(gt_right.cuda())

        up_mask = Variable(up_mask.cuda())
        left_mask = Variable(left_mask.cuda())
        right_mask = Variable(right_mask.cuda())

        up = torch.cat([oppose_up,up_mask_image],1)
        left = torch.cat([oppose_left,left_mask_image],1)
        right = torch.cat([oppose_right,right_mask_image],1)

        #train D
        # D_real = D(gt_up,gt_left,gt_right).squeeze()
        # D_real_loss = criterionBCE(D_real, Variable(torch.ones(D_real.size()).cuda()))
        #
        # #注意力机制版
        # #up_G_result,left_G_result,right_G_result = Mutil_Generator(up,left,right)
        # # z = torch.rand(batch_size, 256, 16, 16).cuda()
        # # up_e4, _ = up_G(torch.cat([oppose_up, inp_up], 1), z)
        # # left_e4, _ = left_G(torch.cat([oppose_left, inp_left], 1), z)
        # # right_e4, _ = right_G(torch.cat([oppose_right, inp_right], 1), z)
        # #
        # # # 图卷积换掉
        # # z = up_e4 + left_e4 + right_e4
        #
        # up_G_result = up_G(up)
        # left_G_result = left_G(left)
        # right_G_result = right_G(right)
        #
        #
        # D_fake = D(up_G_result,left_G_result,right_G_result).squeeze()
        # D_fake_loss = criterionBCE(D_fake, Variable(torch.zeros(D_fake.size()).cuda()))
        #
        # D_train_loss = (D_fake_loss + D_real_loss) * 0.5
        # D_real_acu = torch.ge(D_real, 0.5).float()
        # D_fake_acu = torch.le(D_fake, 0.5).float()
        # D_total_acu = torch.mean(torch.cat((D_real_acu, D_fake_acu), 0))
        # if D_total_acu <= 0.8:
        #     D_train_loss.backward()
        #     D_optimizer.step()
        #     train_hist['D_losses'].append(D_train_loss.item())
        #     D_losses.append(D_train_loss.item())

        #train up_D
        up_D.zero_grad()
        up_D_real = up_D(torch.cat([torch.cat([oppose_up,up_mask_image],1),gt_up],1))
        up_D_real_loss = up_D_real.mean()

        up_G_result, _, _ = Mutil_Generator(up, left, right,up_mask,left_mask,right_mask)
        #print(up_G_result.shape)





        up_D_fake = up_D(torch.cat([torch.cat([oppose_up,up_mask_image],1),up_G_result],1))
        up_D_fake_loss = up_D_fake.mean()

        up_gradient_penalty = calculate_gradient_penalty(up_D,torch.cat([torch.cat([oppose_up,up_mask_image],1),gt_up],1),torch.cat([torch.cat([oppose_up,up_mask_image],1),up_G_result],1))

        up_D_train_loss = up_D_fake_loss-up_D_real_loss+up_gradient_penalty

        up_D_train_loss.backward()
        up_D_optimizer.step()
        train_hist['up_D_losses'].append(up_D_train_loss.item())
        up_D_losses.append(up_D_train_loss.item())

        # train left_D
        left_D.zero_grad()
        left_D_real = left_D(torch.cat([torch.cat([oppose_left, left_mask_image], 1), gt_left],1))
        left_D_real_loss = left_D_real.mean()

        _, left_G_result, _ = Mutil_Generator(up, left, right,up_mask,left_mask,right_mask)



        left_D_fake = left_D(torch.cat([torch.cat([oppose_left, left_mask_image], 1), left_G_result],1))
        left_D_fake_loss = left_D_fake.mean()

        left_gradient_penalty = calculate_gradient_penalty(left_D,torch.cat([torch.cat([oppose_left, left_mask_image], 1), gt_left],1),torch.cat([torch.cat([oppose_left, left_mask_image], 1), left_G_result],1))

        left_D_train_loss = left_D_fake_loss - left_D_real_loss + left_gradient_penalty
        # left_D_real_acu = torch.ge(left_D_real, 0.5).float()
        # left_D_fake_acu = torch.le(left_D_fake, 0.5).float()
        # left_D_total_acu = torch.mean(torch.cat((left_D_real_acu, left_D_fake_acu), 0))
        # if left_D_total_acu <= 0.8:
        left_D_train_loss.backward()
        left_D_optimizer.step()
        train_hist['left_D_losses'].append(left_D_train_loss.item())
        left_D_losses.append(left_D_train_loss.item())

        # train right_D
        right_D.zero_grad()
        right_D_real = right_D(torch.cat([torch.cat([oppose_right, right_mask_image], 1), gt_right],1))
        right_D_real_loss = right_D_real.mean()

        _, _, right_G_result = Mutil_Generator(up, left, right,up_mask,left_mask,right_mask)


        right_D_fake = right_D(torch.cat([torch.cat([oppose_right, right_mask_image], 1), right_G_result],1))
        right_D_fake_loss = right_D_fake.mean()

        right_gradient_penalty = calculate_gradient_penalty(right_D, torch.cat([torch.cat([oppose_right, right_mask_image], 1), gt_right],1), torch.cat([torch.cat([oppose_right, right_mask_image], 1), right_G_result],1))

        right_D_train_loss = right_D_fake_loss - right_D_real_loss + right_gradient_penalty
        # right_D_real_acu = torch.ge(right_D_real, 0.5).float()
        # right_D_fake_acu = torch.le(right_D_fake, 0.5).float()
        # right_D_total_acu = torch.mean(torch.cat((right_D_real_acu, right_D_fake_acu), 0))
        # if right_D_total_acu <= 0.8:
        right_D_train_loss.backward()
        right_D_optimizer.step()
        train_hist['right_D_losses'].append(right_D_train_loss.item())
        right_D_losses.append(right_D_train_loss.item())


        #train local
        up_LD.zero_grad()
        up_real_local = gt_up[:,:,64:192,64:192]
        #up_fake = up_G(up)


        up_fake, _, _ = Mutil_Generator(up, left, right,up_mask,left_mask,right_mask)

        up_fake_local = up_fake[:,:,64:192,64:192]
        up_LD_real = up_LD(up_real_local)
        up_LD_real_loss = up_LD_real.mean()
        up_LD_fake = up_LD(up_fake_local)
        up_LD_fake_loss = up_LD_fake.mean()
        upL_gradient_penalty = calculate_gradient_penalty(up_LD,up_real_local,up_fake_local)

        up_LD_train_loss = up_LD_fake_loss - up_LD_real_loss + upL_gradient_penalty
        # up_LD_real_acu = torch.ge(up_LD_real, 0.5).float()
        # up_LD_fake_acu = torch.le(up_LD_fake, 0.5).float()
        # up_LD_total_acu = torch.mean(torch.cat((up_LD_real_acu, up_LD_fake_acu), 0))
        # if up_LD_total_acu<=0.8:
        up_LD_train_loss.backward()
        up_LD_optimizer.step()
        up_LD_losses.append(up_LD_train_loss.item())

        left_LD.zero_grad()
        left_real_local = gt_left[:, :, 64:192, 64:192]
        #left_fake = left_G(left)
        _, left_fake, _ = Mutil_Generator(up, left, right,up_mask,left_mask,right_mask)



        left_fake_local = left_fake[:, :, 64:192, 64:192]
        left_LD_real = left_LD(left_real_local)
        left_LD_real_loss = left_LD_real.mean()
        left_LD_fake = left_LD(left_fake_local)
        left_LD_fake_loss = left_LD_fake.mean()
        leftL_gradient_penalty = calculate_gradient_penalty(left_LD, left_real_local, left_fake_local)

        left_LD_train_loss = left_LD_fake_loss - left_LD_real_loss + leftL_gradient_penalty
        # left_LD_real_acu = torch.ge(left_LD_real, 0.5).float()
        # left_LD_fake_acu = torch.le(left_LD_fake, 0.5).float()
        # left_LD_total_acu = torch.mean(torch.cat((left_LD_real_acu, left_LD_fake_acu), 0))
        # if left_LD_total_acu <= 0.8:
        left_LD_train_loss.backward()
        left_LD_optimizer.step()
        left_LD_losses.append(left_LD_train_loss.item())

        right_LD.zero_grad()
        right_real_local = gt_right[:, :, 64:192, 64:192]
        #right_fake = right_G(right)



        _,_, right_fake =  Mutil_Generator(up, left, right,up_mask,left_mask,right_mask)

        right_fake_local = right_fake[:, :, 64:192, 64:192]
        right_LD_real = right_LD(right_real_local)
        right_LD_real_loss = right_LD_real.mean()
        right_LD_fake = right_LD(right_fake_local)
        right_LD_fake_loss = right_LD_fake.mean()
        rightL_gradient_penalty = calculate_gradient_penalty(right_LD, right_real_local, right_fake_local)

        right_LD_train_loss = right_LD_fake_loss - right_LD_real_loss + rightL_gradient_penalty
        # right_LD_real_acu = torch.ge(right_LD_real, 0.5).float()
        # right_LD_fake_acu = torch.le(right_LD_fake, 0.5).float()
        # right_LD_total_acu = torch.mean(torch.cat((right_LD_real_acu, right_LD_fake_acu), 0))
        # if right_LD_total_acu <= 0.8:
        right_LD_train_loss.backward()
        right_LD_optimizer.step()
        right_LD_losses.append(right_LD_train_loss.item())


        #train Generator
        Mutil_Generator.zero_grad()

        up_G_result,left_G_result,right_G_result = Mutil_Generator(up, left, right,up_mask,left_mask,right_mask)

        up_D_fake = up_D(torch.cat([torch.cat([oppose_up, up_mask_image], 1), up_G_result], 1))
        up_D_fake_loss = up_D_fake.mean()
        up_fake_local_G = up_G_result[:, :, 64:192, 64:192]
        up_LD_fake = up_LD(up_fake_local_G)
        up_LD_fake_loss = up_LD_fake.mean()
        left_D_fake = left_D(torch.cat([torch.cat([oppose_left, left_mask_image], 1), left_G_result], 1))
        left_D_fake_loss = left_D_fake.mean()
        left_fake_local_G = left_G_result[:, :, 64:192, 64:192]
        left_LD_fake = left_LD(left_fake_local_G)
        left_LD_fake_loss = left_LD_fake.mean()
        right_fake_local_G = right_G_result[:, :, 64:192, 64:192]
        right_LD_fake = right_LD(right_fake_local_G)
        right_LD_fake_loss = right_LD_fake.mean()
        right_D_fake = right_D(torch.cat([torch.cat([oppose_right, right_mask_image], 1), right_G_result], 1))
        right_D_fake_loss = right_D_fake.mean()


        Gro_fake = Gro(up_G_result)
        Gro_real = Gro(gt_up)


        loss_PG = criterionPG(torch.cat([up_G_result,up_G_result,up_G_result],dim = 1),torch.cat([gt_up,gt_up,gt_up],dim = 1))+criterionPG(torch.cat([left_G_result,left_G_result,left_G_result],dim = 1),torch.cat([gt_left,gt_left,gt_left],dim = 1))+criterionPG(torch.cat([right_G_result,right_G_result,right_G_result],dim = 1),torch.cat([gt_right,gt_right,gt_right],dim = 1))

        up_con_loss = consistent_loss_up(up_G_result, left_G_result, right_G_result)
        mutil_G_train_loss = -1*(up_D_fake_loss+left_D_fake_loss+right_D_fake_loss)-(up_LD_fake_loss+left_LD_fake_loss+right_LD_fake_loss)+L1_lambda*(criterionL1(up_G_result,gt_up) + criterionL1(left_G_result,gt_left)+ criterionL1(right_G_result, gt_right))+10*up_con_loss+Lpg_lambda*loss_PG + Lgro_lambda*criterionL1(Gro_fake,Gro_real)###
        mutil_G_train_loss.backward()
        Mutil_G_optimizer.step()
        train_hist['G_losses'].append(mutil_G_train_loss.item())
        w_loss = -1*(up_D_fake_loss+left_D_fake_loss+right_D_fake_loss)-(up_LD_fake_loss+left_LD_fake_loss+right_LD_fake_loss)
        G_w_losses.append(w_loss.item())
        l1_loss = L1_lambda*(criterionL1(up_G_result,gt_up) + criterionL1(left_G_result,gt_left)+ criterionL1(right_G_result, gt_right))
        gro_loss = Lgro_lambda*criterionL1(Gro_fake,Gro_real)
        G_l1_losses.append(l1_loss.item())
        G_losses.append(mutil_G_train_loss.item())
        consis_loss = 10*up_con_loss
        G_consis_losses.append(consis_loss.item())
        G_gro_losses.append(gro_loss.item())
        #save_inter(up_mask_image, left_mask_image, right_mask_image, up_G_result, left_G_result, right_G_result, gt_up,gt_left, gt_right, up_mask, left_mask, right_mask, output_image, epoch,count)
        #count += batch_size

    mutil_G_losses_mean = sum(G_losses)/len(G_losses)
    G_l1_losses_mean = sum(G_l1_losses)/len(G_l1_losses)
    G_w_losses_mean = sum(G_w_losses)/len(G_w_losses)
    G_consis_losses_mean = sum(G_consis_losses)/len(G_consis_losses)
    G_gro_losses_mean = sum(G_gro_losses)/len(G_gro_losses)

    up_D_losses_mean = sum(up_D_losses)/len(up_D_losses)
    left_D_losses_mean = sum(left_D_losses) / len(left_D_losses)
    right_D_losses_mean = sum(right_D_losses) / len(right_D_losses)

    up_LD_losses_mean = sum(up_LD_losses) / len(up_LD_losses)
    left_LD_losses_mean = sum(left_LD_losses) / len(left_LD_losses)
    right_LD_losses_mean = sum(right_LD_losses) / len(right_LD_losses)


    D_losses_mean = 0.5*up_D_losses_mean+0.25*left_D_losses_mean+0.25*right_D_losses_mean#sum(D_losses)/len(D_losses)

    now_rmse = validate_multi(val_loader,Mutil_Generator)


    if now_rmse<best_loss:
        best_loss = now_rmse
        torch.save({'epoch':epoch,
                    'model':Mutil_Generator.state_dict(),
                    'now_rmse':now_rmse
                    }, os.path.join(output_dir, 'best_generator.pth'))
        #torch.save(D.state_dict(), os.path.join(output_dir, 'best_discriminator.pth'))
        torch.save(up_D.state_dict(), os.path.join(output_dir, 'best_up_discriminator.pth'))
        torch.save(left_D.state_dict(), os.path.join(output_dir, 'best_left_discriminator.pth'))
        torch.save(right_D.state_dict(), os.path.join(output_dir, 'best_right_discriminator.pth'))
        torch.save(up_LD.state_dict(), os.path.join(output_dir, 'best_up_localdiscriminator.pth'))
        torch.save(left_LD.state_dict(), os.path.join(output_dir, 'best_left_localdiscriminator.pth'))
        torch.save(right_LD.state_dict(), os.path.join(output_dir, 'best_right_localdiscriminator.pth'))

    if epoch  % save_interv == 0:
        torch.save({'epoch':epoch,
                    'model':Mutil_Generator.state_dict(),
                    #'model': Mutil_Generator.state_dict(),
                    'G_loss':mutil_G_losses_mean
                    }, os.path.join(output_dir, '{}_generator.pth'.format(epoch+1)))
        #torch.save(D.state_dict(), os.path.join(output_dir, '{}_discriminator.pth'.format(epoch+1)))
        save_inter(up_mask_image,left_mask_image,right_mask_image,up_G_result,left_G_result,right_G_result,gt_up,gt_left,gt_right,up_mask,left_mask,right_mask,output_image,epoch)
        torch.save(up_D.state_dict(), os.path.join(output_dir, '{}_up_discriminator.pth'.format(epoch + 1)))
        torch.save(left_D.state_dict(), os.path.join(output_dir, '{}_left_discriminator.pth'.format(epoch + 1)))
        torch.save(right_D.state_dict(), os.path.join(output_dir, '{}_right_discriminator.pth'.format(epoch + 1)))
        torch.save(up_LD.state_dict(), os.path.join(output_dir, '{}_up_localdiscriminator.pth'.format(epoch + 1)))
        torch.save(left_LD.state_dict(), os.path.join(output_dir, '{}_left_localdiscriminator.pth'.format(epoch + 1)))
        torch.save(right_LD.state_dict(), os.path.join(output_dir, '{}_right_localdiscriminator.pth'.format(epoch + 1)))
    print("epoch-{};,D_up_loss : {:.4}, D_left_loss : {:.4},D_right_loss : {:.4},LD_up_loss : {:.4}, LD_left_loss : {:.4},LD_right_loss : {:.4},mutil_G_loss : {:.4},now_mrse:{:.4} ".format(epoch+1,up_D_losses_mean,left_D_losses_mean,right_D_losses_mean,up_LD_losses_mean,left_LD_losses_mean,right_LD_losses_mean,mutil_G_losses_mean,now_rmse))#
    print("l1_loss:{:.4},w_loss:{:.4},consis_loss:{:.4},gro_loss:{:.4}".format(G_l1_losses_mean,G_w_losses_mean,G_consis_losses_mean,G_gro_losses_mean))#






