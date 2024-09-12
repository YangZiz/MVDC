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

batch_size = 1
ROOT = r'/yy/data/depth_map/multi_view'
val_root = r'/yy/data/depth_map/multi_view'
output_dir = r'/yy/code/pix2pix/output/checkpoint/3_3'
gro_checkpoint = r'/yy/code/pix2pix/output/checkpoint/gronet/best_generator.pth'
output_image = r'/yy/code/pix2pix/output/depth_map/3_3'
#pretrain_path = r'/yy/code/pix2pix/output/checkpoint/pretrain/pretrain_best_model.pth'
epochs = 500
L1_lambda = 100
Lgro_lambda = 200
Lpg_lambda = 50
save_interv = 20
Lconstra_lambda = 100
train_dataset = Dataset('train',ROOT)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
val_dataset = val_Dataset('val',val_root)
val_loader = DataLoader(val_dataset,batch_size=8,shuffle=False)


Mutil_Generator = Mutil_view_PGenerator()


Globle_Discriminator = Multi_Globle_Discriminator()
Local_Discriminator = Multi_Local_Discriminator()
Mutil_D = WDiscriminator(in_channels=3)


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
    Globle_Discriminator.load_state_dict(os.path.join(output_dir,'best_globle_discriminator.pth'))
    Local_Discriminator.load_state_dict(os.path.join(output_dir,'best_local_discriminator.pth'))



else:
    Mutil_Generator.weight_init(mean=0.0,std=0.02)
    Mutil_D.weight_init(mean = 0.0,std=0.02)

    Globle_Discriminator.weight_init(mean=0.0,std = 0.02)
    Local_Discriminator.weight_init(mean=0.0,std = 0.02)

print(best_loss)
criterionL1 = nn.L1Loss().cuda()
criterionMSE = nn.MSELoss().cuda()
criterionBCE = nn.BCELoss().cuda()

consistent_loss_up = Consistent_loss_up_4()


loss_func = nn.MSELoss()
criterionPG = PerceptualLoss(loss_func,layer_indexs=[3,8,15,22])

Mutil_Generator.cuda()
Mutil_D.cuda()
Globle_Discriminator.cuda()
Local_Discriminator.cuda()
Gro.cuda()
Gro.load_state_dict(torch.load(gro_checkpoint))
Gro.eval()

Mutil_Generator.train()
Mutil_D.train()
Globle_Discriminator.train()
Local_Discriminator.train()

Mutil_G_optimizer = optim.Adam(Mutil_Generator.parameters(),lr=0.00002,betas=(0.5,0.999))
Globle_Discriminator_optimizer = optim.Adam(Globle_Discriminator.parameters(),lr = 0.000006,betas=(0.5,0.999))
Local_Discriminator_optimizer = optim.Adam(Local_Discriminator.parameters(),lr = 0.000006,betas=(0.5,0.999))
Mutil_D_optimizer = optim.Adam(Mutil_D.parameters(),lr = 0.000006,betas=(0.5,0.999))


train_hist = {}

train_hist['G_losses'] = []
train_hist['up_D_losses'] = []
train_hist['left_D_losses'] = []
train_hist['right_D_losses'] = []
#train_hist['D_losses'] = []
train_hist['up_LD_losses'] = []
train_hist['left_LD_losses'] = []
train_hist['right_LD_losses'] = []
train_hist['Mutil_D_losses'] = []

for epoch in range(now_epoch,epochs):
    print('Epoch [%d/%d]' % (epoch + 1, epochs))
    #D_losses = []

    G_losses = []
    G_w_losses = []
    G_l1_losses = []
    G_adv_losses = []
    G_consis_losses = []
    G_gro_losses = []
    G_constra_losses = []

    Mutil_D_losses = []


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
    y = 0
    #count = 0

    for data in train_loader:
        #D.zero_grad()

        oppose_up = torch.cat([data['up']['oppose'],data['up']['oppose']],0)
        inp_up = data['up']['inp']
        gt_up = torch.cat([data['up']['gt'],data['up']['gt']],0)
        rio_up = data['up']['rio']
        mask_up = data['up']['mask']
        mask_add = torch.ones((batch_size*2, 1, 256, 256), dtype=torch.float32)
        #up_mask_image1,up_mask1 = add_mask(inp_up,rio_up)
        up_mask_image1 = inp_up.clone()
        up_mask_image1[mask_up == 0] = 0
        up_mask1 = mask_up.clone()
        up_mask_image2, up_mask2 = add_mask(inp_up, rio_up)
        #inp_up = torch.cat([data['up']['inp'],data['up']['inp']],0)
        up_mask_image = torch.cat([up_mask_image1,up_mask_image2],0)
        up_mask = torch.cat([up_mask1,up_mask2],0)
        oppose_up = np.array(oppose_up)
        up_mask_image = np.array(up_mask_image)
        up_mask = np.array(up_mask)
        gt_up = np.array(gt_up)
        oppose_up = transforms.ToTensor()(oppose_up).permute(1, 0, 2).unsqueeze(1)
        gt_up = transforms.ToTensor()(gt_up).permute(1, 0, 2).unsqueeze(1)

        up_mask_image = transforms.ToTensor()(up_mask_image).permute(1, 0, 2).unsqueeze(1)
        up_mask = transforms.ToTensor()(up_mask).permute(1, 0, 2).unsqueeze(1)
        up_mask = torch.cat((mask_add, up_mask), axis=1)

        oppose_left = torch.cat([data['left']['oppose'],data['left']['oppose']],0)
        inp_left = data['left']['inp']
        gt_left = torch.cat([data['left']['gt'],data['left']['gt']],0)
        rio_left = data['left']['rio']
        #left_mask_image1,left_mask1 = add_mask(inp_left, rio_left)
        mask_left = data['left']['mask']
        left_mask_image1 = inp_left.clone()
        left_mask_image1[mask_left == 0] = 0
        left_mask1 = mask_left.clone()
        left_mask_image2, left_mask2 = add_mask(inp_left, rio_left)
        #inp_left = torch.cat([data['left']['inp'],data['left']['inp']],0)
        left_mask_image = torch.cat([left_mask_image1, left_mask_image2], 0)
        left_mask = torch.cat([left_mask1, left_mask2], 0)
        oppose_left = np.array(oppose_left)
        left_mask_image = np.array(left_mask_image)
        left_mask = np.array(left_mask)
        gt_left = np.array(gt_left)
        oppose_left = transforms.ToTensor()(oppose_left).permute(1, 0, 2).unsqueeze(1)
        left_mask_image = transforms.ToTensor()(left_mask_image).permute(1, 0, 2).unsqueeze(1)
        gt_left = transforms.ToTensor()(gt_left).permute(1, 0, 2).unsqueeze(1)
        left_mask = transforms.ToTensor()(left_mask).permute(1, 0, 2).unsqueeze(1)
        left_mask = torch.cat((mask_add, left_mask), axis=1)

        oppose_right = torch.cat([data['right']['oppose'],data['right']['oppose']],0)
        inp_right = data['right']['inp']
        gt_right = torch.cat([data['right']['gt'],data['right']['gt']],0)
        rio_right = data['right']['rio']
        #right_mask_image1,right_mask1 = add_mask(inp_right, rio_right)
        mask_right = data['right']['mask']
        right_mask_image1 = inp_right.clone()
        right_mask_image1[mask_right == 0] = 0
        right_mask1 = mask_right.clone()
        right_mask_image2, right_mask2 = add_mask(inp_right, rio_right)
        #inp_right = torch.cat([data['right']['inp'], data['right']['inp']], 0)
        right_mask_image = torch.cat([right_mask_image1, right_mask_image2], 0)
        right_mask = torch.cat([right_mask1, right_mask2], 0)
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


        #train Mutil_D
        Mutil_D.zero_grad()
        Mutil_real = Mutil_D(torch.cat([gt_left,gt_up,gt_right],1))
        up_G_result, left_G_result, right_G_result = Mutil_Generator(up, left, right, up_mask, left_mask, right_mask)
        Mutil_fake = Mutil_D(torch.cat([left_G_result,up_G_result,right_G_result],1))
        Mutil_D_real_loss = Mutil_real.mean()
        Mutil_D_fake_loss = Mutil_fake.mean()
        Mutil_gradient_penalty = calculate_gradient_penalty(Mutil_D,torch.cat([gt_left,gt_up,gt_right],1),torch.cat([left_G_result,up_G_result,right_G_result],1))
        Mutil_D_train_loss = Mutil_D_fake_loss-Mutil_D_real_loss + Mutil_gradient_penalty
        Mutil_D_losses.append(Mutil_D_train_loss.item())
        train_hist['Mutil_D_losses'].append(Mutil_D_train_loss.item())
        Mutil_D_train_loss.backward()
        Mutil_D_optimizer.step()


        #train Globle Discriminator
        Globle_Discriminator.zero_grad()
        up_D_real,left_D_real,right_D_real = Globle_Discriminator(torch.cat([torch.cat([oppose_up,up_mask_image],1),gt_up],1),torch.cat([torch.cat([oppose_left, left_mask_image], 1), gt_left],1),torch.cat([torch.cat([oppose_right, right_mask_image], 1), gt_right],1))
        up_G_result,left_G_result,right_G_result = Mutil_Generator(up, left, right,up_mask,left_mask,right_mask)
        up_D_fake,left_D_fake,right_D_fake = Globle_Discriminator(torch.cat([torch.cat([oppose_up,up_mask_image],1),up_G_result],1),torch.cat([torch.cat([oppose_left, left_mask_image], 1), left_G_result],1),torch.cat([torch.cat([oppose_right, right_mask_image], 1), right_G_result],1))
        up_D_real_loss = up_D_real.mean()
        left_D_real_loss = left_D_real.mean()
        right_D_real_loss = right_D_real.mean()

        up_D_fake_loss = up_D_fake.mean()
        left_D_fake_loss = left_D_fake.mean()
        right_D_fake_loss = right_D_fake.mean()

        up_gradient_penalty,left_gradient_penalty,right_gradient_penalty = calculate_gradient_penalty_Multi(Globle_Discriminator,torch.cat([torch.cat([oppose_up, up_mask_image], 1), gt_up],1), torch.cat([torch.cat([oppose_left, left_mask_image], 1), gt_left],1),torch.cat([torch.cat([oppose_right, right_mask_image], 1), gt_right],1),torch.cat([torch.cat([oppose_up, up_mask_image], 1), up_G_result], 1),torch.cat([torch.cat([oppose_left, left_mask_image], 1), left_G_result], 1),torch.cat([torch.cat([oppose_right, right_mask_image], 1), right_G_result], 1))

        up_D_train_loss = up_D_fake_loss - up_D_real_loss + up_gradient_penalty
        left_D_train_loss = left_D_fake_loss - left_D_real_loss + left_gradient_penalty
        right_D_train_loss = right_D_fake_loss - right_D_real_loss + right_gradient_penalty
        up_D_losses.append(up_D_train_loss.item())
        left_D_losses.append(left_D_train_loss.item())
        right_D_losses.append(right_D_train_loss.item())

        train_hist['up_D_losses'].append(up_D_train_loss.item())
        train_hist['left_D_losses'].append(left_D_train_loss.item())
        train_hist['right_D_losses'].append(right_D_train_loss.item())
        D_loss = up_D_train_loss+left_D_train_loss+right_D_train_loss
        D_loss.backward()
        Globle_Discriminator_optimizer.step()

        #train_Local_discriminator
        Local_Discriminator.zero_grad()
        up_real_local = gt_up[:, :, 64:192, 64:192]
        left_real_local = gt_left[:, :, 64:192, 64:192]
        right_real_local = gt_right[:, :, 64:192, 64:192]


        # up_fake = up_G(up)

        up_fake, left_fake, right_fake = Mutil_Generator(up, left, right, up_mask, left_mask, right_mask)

        up_fake_local = up_fake[:, :, 64:192, 64:192]
        left_fake_local = left_fake[:,:,64:192,64:192]
        right_fake_local = right_fake[:, :, 64:192, 64:192]

        up_LD_real,left_LD_real,right_LD_real = Local_Discriminator(up_real_local,left_real_local,right_real_local)
        up_LD_fake,left_LD_fake,right_LD_fake = Local_Discriminator(up_fake_local,left_fake_local,right_fake_local)

        up_LD_real_loss = up_LD_real.mean()
        left_LD_real_loss = left_LD_real.mean()
        right_LD_real_loss = right_LD_real.mean()

        up_LD_fake_loss = up_LD_fake.mean()
        left_LD_fake_loss = left_LD_fake.mean()
        right_LD_fake_loss = right_LD_fake.mean()

        upL_gradient_penalty,leftL_gradient_penalty,rightL_gradient_penalty = calculate_gradient_penalty_Multi(Local_Discriminator, up_real_local,left_real_local,right_real_local, up_fake_local,left_fake_local,right_fake_local)

        up_LD_train_loss = up_LD_fake_loss - up_LD_real_loss + upL_gradient_penalty
        left_LD_train_loss = left_LD_fake_loss - left_LD_real_loss + leftL_gradient_penalty
        right_LD_train_loss = right_LD_fake_loss - right_LD_real_loss + rightL_gradient_penalty

        up_LD_losses.append(up_LD_train_loss.item())
        left_LD_losses.append(left_LD_train_loss.item())
        right_LD_losses.append(right_LD_train_loss.item())
        LD_loss = up_LD_train_loss+left_LD_train_loss+right_LD_train_loss
        LD_loss.backward()
        Local_Discriminator_optimizer.step()





        #train Generator
        Mutil_Generator.zero_grad()

        up_G_result_G,left_G_result_G,right_G_result_G = Mutil_Generator(up, left, right,up_mask,left_mask,right_mask)

        up_D_fake_G,left_D_fake_G,right_D_fake_G = Globle_Discriminator(torch.cat([torch.cat([oppose_up,up_mask_image],1),up_G_result_G],1),torch.cat([torch.cat([oppose_left, left_mask_image], 1), left_G_result_G],1),torch.cat([torch.cat([oppose_right, right_mask_image], 1), right_G_result_G],1))

        Mutil_fake_G = Mutil_D(torch.cat([left_G_result_G, up_G_result_G, right_G_result_G], 1))
        up_fake_local_G = up_G_result_G[:, :, 64:192, 64:192]
        left_fake_local_G = left_G_result_G[:, :, 64:192, 64:192]
        right_fake_local_G = right_G_result_G[:, :, 64:192, 64:192]
        up_LD_fake_G,left_LD_fake_G,right_LD_fake_G = Local_Discriminator(up_fake_local_G,left_fake_local_G,right_fake_local_G)

        Mutil_D_fake_loss_G = Mutil_fake_G.mean()
        up_D_fake_loss_G = up_D_fake_G.mean()
        up_LD_fake_loss_G = up_LD_fake_G.mean()
        left_D_fake_loss_G = left_D_fake_G.mean()
        left_LD_fake_loss_G = left_LD_fake_G.mean()
        right_LD_fake_loss_G = right_LD_fake_G.mean()
        right_D_fake_loss_G = right_D_fake_G.mean()



        Gro_fake = Gro(up_G_result_G)
        Gro_real = Gro(gt_up)

        contra_loss = criterionL1(up_G_result_G[0],up_G_result_G[1]) + criterionL1(left_G_result_G[0],left_G_result_G[1]) + criterionL1(left_G_result_G[0],left_G_result_G[1])


        loss_PG = criterionPG(torch.cat([up_G_result_G,up_G_result_G,up_G_result_G],dim = 1),torch.cat([gt_up,gt_up,gt_up],dim = 1))+criterionPG(torch.cat([left_G_result_G,left_G_result_G,left_G_result_G],dim = 1),torch.cat([gt_left,gt_left,gt_left],dim = 1))+criterionPG(torch.cat([right_G_result_G,right_G_result_G,right_G_result_G],dim = 1),torch.cat([gt_right,gt_right,gt_right],dim = 1))

        up_con_loss = consistent_loss_up(up_G_result_G, left_G_result_G, right_G_result_G)
        mutil_G_train_loss = -1*Mutil_D_fake_loss_G - (up_D_fake_loss_G+left_D_fake_loss_G+right_D_fake_loss_G)-(up_LD_fake_loss_G+left_LD_fake_loss_G+right_LD_fake_loss_G)+L1_lambda*(criterionL1(up_G_result_G,gt_up) + criterionL1(left_G_result_G,gt_left)+ criterionL1(right_G_result_G, gt_right))+10*up_con_loss+Lpg_lambda*loss_PG + Lgro_lambda*criterionL1(Gro_fake,Gro_real) + Lconstra_lambda * contra_loss###
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
        G_constra_losses.append(contra_loss.item())
        # print(y)
        # y += 1

        #save_inter(up_mask_image, left_mask_image, right_mask_image, up_G_result, left_G_result, right_G_result, gt_up,gt_left, gt_right, up_mask, left_mask, right_mask, output_image, epoch,count)
        #count += batch_size

    mutil_G_losses_mean = sum(G_losses)/len(G_losses)
    G_l1_losses_mean = sum(G_l1_losses)/len(G_l1_losses)
    G_w_losses_mean = sum(G_w_losses)/len(G_w_losses)
    G_consis_losses_mean = sum(G_consis_losses)/len(G_consis_losses)
    G_gro_losses_mean = sum(G_gro_losses)/len(G_gro_losses)
    G_constra_losses_mean = sum(G_constra_losses) / len(G_constra_losses)

    Mutil_D_losses_mean = sum(Mutil_D_losses)/len(Mutil_D_losses)
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

        torch.save(Globle_Discriminator.state_dict(), os.path.join(output_dir, 'best_globle_discriminator.pth'))
        torch.save(Local_Discriminator.state_dict(), os.path.join(output_dir, 'best_local_discriminator.pth'))
        torch.save(Mutil_D.state_dict(),os.path.join(output_dir, 'best_mutil_discriminator.pth'))

    if epoch  % save_interv == 0:
        torch.save({'epoch':epoch,
                    'model':Mutil_Generator.state_dict(),
                    #'model': Mutil_Generator.state_dict(),
                    'now_rmse': now_rmse,
                    'G_loss':mutil_G_losses_mean
                    }, os.path.join(output_dir, '{}_generator.pth'.format(epoch+1)))
        #torch.save(D.state_dict(), os.path.join(output_dir, '{}_discriminator.pth'.format(epoch+1)))
        save_inter(up_mask_image,left_mask_image,right_mask_image,up_G_result,left_G_result,right_G_result,gt_up,gt_left,gt_right,up_mask,left_mask,right_mask,output_image,epoch)
        torch.save(Globle_Discriminator.state_dict(), os.path.join(output_dir, '{}_globle_discriminator.pth'.format(epoch + 1)))
        torch.save(Local_Discriminator.state_dict(), os.path.join(output_dir, '{}_local_discriminator.pth'.format(epoch + 1)))
        torch.save(Mutil_D.state_dict(), os.path.join(output_dir, '{}_mutil_discriminator.pth'.format(epoch + 1)))

    print("epoch-{};Mutil_D_loss : {:.4},D_up_loss : {:.4}, D_left_loss : {:.4},D_right_loss : {:.4},LD_up_loss : {:.4}, LD_left_loss : {:.4},LD_right_loss : {:.4},mutil_G_loss : {:.4},now_mrse:{:.4} ".format(epoch+1,Mutil_D_losses_mean,up_D_losses_mean,left_D_losses_mean,right_D_losses_mean,up_LD_losses_mean,left_LD_losses_mean,right_LD_losses_mean,mutil_G_losses_mean,now_rmse))#
    print("l1_loss:{:.4},w_loss:{:.4},consis_loss:{:.4},gro_loss:{:.4},constra_loss:{:.4}".format(G_l1_losses_mean,G_w_losses_mean,G_consis_losses_mean,G_gro_losses_mean,G_constra_losses_mean))#






