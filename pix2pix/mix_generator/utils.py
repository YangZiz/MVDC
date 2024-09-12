import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg16
import warnings
import numpy as np
from PIL import Image, ImageDraw
import math
import random
import os
from torch import autograd
from torch.nn import functional as F
from torchvision import models

#warnings.filterwarnings('ignore')


# 计算特征提取模块的感知损失
def vgg16_loss(feature_module,loss_func,y,y_):
    out=feature_module(y)
    out_=feature_module(y_)
    loss=loss_func(out,out_)
    return loss

# 获取指定的特征提取模块
# def get_feature_module(layer_index,device=None):
#     vgg = vgg16(pretrained=True, progress=True).features
#     vgg.cuda()
#     vgg.eval()
#
#     # 冻结参数
#     for parm in vgg.parameters():
#         parm.requires_grad = False
#
#     feature_module = vgg[0:layer_index + 1]
#     feature_module.to(device)
#     return feature_module
#
#
# # 计算指定的组合模块的感知损失
# class PerceptualLoss(nn.Module):
#     def __init__(self,loss_func,layer_indexs=None,device=None):
#         super(PerceptualLoss, self).__init__()
#         self.creation=loss_func
#         self.layer_indexs=layer_indexs
#         self.device=device
#
#     def forward(self,y,y_):
#         loss=0
#         for index in self.layer_indexs:
#             feature_module=get_feature_module(index,self.device)
#             loss+=vgg16_loss(feature_module,self.creation,y,y_)
#         return loss

class PerceptualLoss(nn.Module):
    def __init__(self, loss_func, layer_indexs=None, device=None):
        super(PerceptualLoss, self).__init__()
        self.creation = loss_func
        self.layer_indexs = layer_indexs
        self.device = device
        self.feature_modules = [self.get_feature_module(index) for index in layer_indexs]

    def get_feature_module(self, layer_index):
        vgg = vgg16(pretrained=True, progress=True).features
        vgg.cuda()
        vgg.to(self.device)
        vgg.eval()

        # 冻结参数
        for param in vgg.parameters():
            param.requires_grad = False

        feature_module = vgg[0:layer_index + 1]
        return feature_module

    def forward(self, y, y_):
        loss = 0
        for feature_module in self.feature_modules:
            loss += vgg16_loss(feature_module, self.creation, y, y_)
        return loss


def RandomBrush(
    max_tries,
    s,
    min_num_vertex = 4,
    max_num_vertex = 18,
    mean_angle = 2*math.pi / 5,
    angle_range = 2*math.pi / 15,
    min_width = 12,
    max_width = 48):
    H, W = s, s
    average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)
    for _ in range(np.random.randint(max_tries)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.uint8)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 0)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 1)
    return mask


def RandomMask(s, hole_range=[0,1]):
    coef = min(hole_range[0] + hole_range[1], 1.0)
    while True:
        mask = np.ones((s, s), np.uint8)
        def Fill(max_size):
            w, h = np.random.randint(max_size), np.random.randint(max_size)
            ww, hh = w // 2, h // 2
            x, y = np.random.randint(-ww, s - w + ww), np.random.randint(-hh, s - h + hh)
            mask[max(y, 0): min(y + h, s), max(x, 0): min(x + w, s)] = 0
        def MultiFill(max_tries, max_size):
            for _ in range(np.random.randint(max_tries)):
                Fill(max_size)
        MultiFill(int(4 * coef), s // 2)
        MultiFill(int(2 * coef), s)
        mask = np.logical_and(mask, 1 - RandomBrush(int(8 * coef), s))  # hole denoted as 0, reserved as 1
        hole_ratio = 1 - np.mean(mask)
        if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
            continue
        return mask[np.newaxis, ...].astype(np.float32)

def add_mask(image_list,rio_list):
    new_image = image_list.clone()
    images_mask = torch.ones(image_list.shape)
    new_images_mask = torch.ones(image_list.shape)
    for i in range(image_list.shape[0]):
        image = image_list[i]
        mask_i = torch.ones(image_list[0].shape)
        new_mask_i = torch.ones(image_list[0].shape)
        rio = rio_list[i]
        nonzero_indices = np.nonzero(rio)
        #print(nonzero_indices[0].max)
        # 获取 x 和 y 的最小值和最大值
        min_x, max_x = torch.min(nonzero_indices[:,0]),torch.max(nonzero_indices[:,0])
        min_y, max_y = torch.min(nonzero_indices[:,1]),torch.max(nonzero_indices[:,1])
        len_s = max(max_x-min_x,max_y-min_y)
        center_x = (min_x+max_x)//2
        center_y = (min_y+max_y)//2
        mask = RandomMask(s=len_s, hole_range=[0.2,0.4])
        mask_t = np.ones(mask.shape)
        mask_t = np.squeeze(mask_t)
        mask = mask * 255
        # print(mask)
        # mask = mask.repeat(3, axis=0)
        mask = mask.transpose(1, 2, 0)
        mask = np.squeeze(mask)
        mask_rio = image[center_x-len_s//2:center_x-len_s//2+len_s,center_y-len_s//2:center_y-len_s//2+len_s].clone()
        mask_renge = rio[center_x-len_s//2:center_x-len_s//2+len_s,center_y-len_s//2:center_y-len_s//2+len_s].clone()
        mask_renge = mask_renge.numpy()
        #print(mask_t.shape,mask.shape,mask_renge.shape,mask_rio.shape)
        mask_rio[(mask == 0) & (mask_renge == 1)] = 0
        mask_t[(mask == 0) & (mask_renge ==1)] = 0
        image[center_x-len_s//2:center_x-len_s//2+len_s,center_y-len_s//2:center_y-len_s//2+len_s] = mask_rio
        new_image[i] = image
        mask_i[center_x-len_s//2:center_x-len_s//2+len_s,center_y-len_s//2:center_y-len_s//2+len_s] = torch.Tensor(mask_t)
        new_mask_i[image == 0] = 0
        new_images_mask[i] = new_mask_i
        images_mask[i] = mask_i

    return new_image,new_images_mask

def save_inter_sig(inp,output,gt,save_root,epoch):
    save_dir = os.path.join(save_root,'{}'.format(epoch))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(inp.shape[0]):
        inp_i_up = inp[i][1]


        output_i_up = output[i]


        gt_i_up = gt[i]


        #print(inp_i_right.shape)
        #print(output_i_right.shape)
        up = inp_i_up.squeeze().cpu().numpy()#.detach()


        predict_up = output_i_up.squeeze().cpu().detach().numpy()



        g_up = gt_i_up.squeeze().cpu().detach().numpy()


        inp_up_i = up * 255
        inp_up_i[inp_up_i < 0] = 0
        inp_up_i[inp_up_i > 255] = 255



        gene_image_up = predict_up * 255
        gene_image_up[gene_image_up < 0] = 0
        gene_image_up[gene_image_up > 255] = 255



        gt_up_i = g_up * 255
        gt_up_i[gt_up_i < 0] = 0
        gt_up_i[gt_up_i > 255] = 255


        image_up = Image.fromarray(inp_up_i.astype(np.uint8),'L')
        image_up.save(os.path.join(save_dir, '{}_inp_up.png'.format(i+1)))




        pre_up = Image.fromarray(gene_image_up.astype(np.uint8), 'L')
        pre_up.save(os.path.join(save_dir, '{}_predict_up.png'.format(i+1)))


        la_up = Image.fromarray(gt_up_i.astype(np.uint8), 'L')
        la_up.save(os.path.join(save_dir, '{}_gt_up.png'.format(i + 1)))

def save_inter(inp_up,inp_left,inp_right,output_up,output_left,output_right,gt_up,gt_left,gt_right,up_mask,left_mask,right_mask,save_root,epoch):
    save_dir = os.path.join(save_root,'{}'.format(epoch))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(inp_up.shape[0]):
        inp_i_up = inp_up[i]
        inp_i_left = inp_left[i]
        inp_i_right = inp_right[i]

        up_i_mask = up_mask[i][1]
        left_i_mask = left_mask[i][1]
        right_i_mask = right_mask[i][1]
        #print(inp_i_right.shape)
        output_i_up = output_up[i]
        output_i_left = output_left[i]
        output_i_right = output_right[i]

        gt_i_up = gt_up[i]
        gt_i_left = gt_left[i]
        gt_i_right = gt_right[i]

        #print(inp_i_right.shape)
        #print(output_i_right.shape)
        up = inp_i_up.squeeze().cpu().numpy()#.detach()
        left = inp_i_left.squeeze().cpu().numpy()#.detach()
        right = inp_i_right.squeeze().cpu().numpy()#.detach()
        # print(up.shape)
        # print(left.shape)
        # print(right.shape)
        predict_up = output_i_up.squeeze().cpu().detach().numpy()
        predict_left = output_i_left.squeeze().cpu().detach().numpy()
        predict_right = output_i_right.squeeze().cpu().detach().numpy()

        up_m = up_i_mask.squeeze().cpu().detach().numpy()
        left_m = left_i_mask.squeeze().cpu().detach().numpy()
        right_m = right_i_mask.squeeze().cpu().detach().numpy()

        g_up = gt_i_up.squeeze().cpu().detach().numpy()
        g_left = gt_i_left.squeeze().cpu().detach().numpy()
        g_right = gt_i_right.squeeze().cpu().detach().numpy()

        inp_up_i = up * 255
        inp_up_i[inp_up_i < 0] = 0
        inp_up_i[inp_up_i > 255] = 255

        inp_left_i = left * 255
        inp_left_i[inp_left_i < 0] = 0
        inp_left_i[inp_left_i > 255] = 255

        inp_right_i = right * 255
        inp_right_i[inp_right_i < 0] = 0
        inp_right_i[inp_right_i > 255] = 255

        gene_image_up = predict_up * 255
        gene_image_up[gene_image_up < 0] = 0
        gene_image_up[gene_image_up > 255] = 255

        gene_image_left = predict_left * 255
        gene_image_left[gene_image_left < 0] = 0
        gene_image_left[gene_image_left > 255] = 255

        gene_image_right = predict_right * 255
        gene_image_right[gene_image_right < 0] = 0
        gene_image_right[gene_image_right > 255] = 255

        gt_up_i = g_up * 255
        gt_up_i[gt_up_i < 0] = 0
        gt_up_i[gt_up_i > 255] = 255

        gt_left_i = g_left * 255
        gt_left_i[gt_left_i < 0] = 0
        gt_left_i[gt_left_i > 255] = 255

        gt_right_i = g_right * 255
        gt_right_i[gt_right_i < 0] = 0
        gt_right_i[gt_right_i > 255] = 255

        up_m = up_m*255
        up_m[up_m<0] = 0
        up_m[up_m>255] = 255

        left_m = left_m * 255
        left_m[left_m < 0] = 0
        left_m[left_m > 255] = 255

        right_m = right_m * 255
        right_m[right_m < 0] = 0
        right_m[right_m > 255] = 255


        image_up = Image.fromarray(inp_up_i.astype(np.uint8),'L')
        image_up.save(os.path.join(save_dir, '{}_inp_up.png'.format(i+1)))

        image_left = Image.fromarray(inp_left_i.astype(np.uint8), 'L')
        image_left.save(os.path.join(save_dir, '{}_inp_left.png'.format(i + 1)))

        image_right = Image.fromarray(inp_right_i.astype(np.uint8), 'L')
        image_right.save(os.path.join(save_dir, '{}_inp_right.png'.format(i + 1)))

        pre_up = Image.fromarray(gene_image_up.astype(np.uint8), 'L')
        pre_up.save(os.path.join(save_dir, '{}_predict_up.png'.format(i+1)))

        pre_left = Image.fromarray(gene_image_left.astype(np.uint8), 'L')
        pre_left.save(os.path.join(save_dir, '{}_predict_left.png'.format(i + 1)))

        pre_right = Image.fromarray(gene_image_right.astype(np.uint8), 'L')
        pre_right.save(os.path.join(save_dir, '{}_predict_right.png'.format(i + 1)))

        la_up = Image.fromarray(gt_up_i.astype(np.uint8), 'L')
        la_up.save(os.path.join(save_dir, '{}_gt_up.png'.format(i + 1)))

        la_left = Image.fromarray(gt_left_i.astype(np.uint8), 'L')
        la_left.save(os.path.join(save_dir, '{}_gt_left.png'.format(i + 1)))

        la_right = Image.fromarray(gt_right_i.astype(np.uint8), 'L')
        la_right.save(os.path.join(save_dir, '{}_gt_right.png'.format(i + 1)))

        m_up = Image.fromarray(up_m.astype(np.uint8), 'L')
        m_up.save(os.path.join(save_dir, '{}_mask_up.png'.format(i + 1)))

        m_left = Image.fromarray(left_m.astype(np.uint8), 'L')
        m_left.save(os.path.join(save_dir, '{}_mask_left.png'.format(i + 1)))

        m_right = Image.fromarray(right_m.astype(np.uint8), 'L')
        m_right.save(os.path.join(save_dir, '{}_mask_right.png'.format(i + 1)))

def calculate_gradient_penalty(D,real_images, fake_images):
    eta = torch.FloatTensor(real_images.shape[0], 1, 1, 1).uniform_(0, 1)
    eta = eta.expand(real_images.shape[0], real_images.size(1), real_images.size(2), real_images.size(3))
    eta = eta.cuda()

    interpolated = eta * real_images + ((1 - eta) * fake_images)

    interpolated = interpolated.cuda()


        # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
    prob_interpolated = D(interpolated)

        # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(
                                      prob_interpolated.size()).cuda(),
                                  create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return grad_penalty

def calculate_gradient_penalty_Multi(D,real_images_up,real_images_left,real_images_right, fake_images_up,fake_images_left,fake_images_right):
    eta = torch.FloatTensor(real_images_up.shape[0], 1, 1, 1).uniform_(0, 1)
    eta = eta.expand(real_images_up.shape[0], real_images_up.size(1), real_images_up.size(2), real_images_up.size(3))
    eta = eta.cuda()

    interpolated_up = eta * real_images_up + ((1 - eta) * fake_images_up)

    interpolated_up = interpolated_up.cuda()

    interpolated_left = eta * real_images_left + ((1 - eta) * fake_images_left)

    interpolated_left = interpolated_left.cuda()

    interpolated_right = eta * real_images_right + ((1 - eta) * fake_images_right)

    interpolated_right = interpolated_right.cuda()


        # define it to calculate gradient
    interpolated_up = Variable(interpolated_up, requires_grad=True)
    interpolated_left = Variable(interpolated_left, requires_grad=True)
    interpolated_right = Variable(interpolated_right, requires_grad=True)

        # calculate probability of interpolated examples
    prob_interpolated_up,prob_interpolated_left,prob_interpolated_right = D(interpolated_up,interpolated_left,interpolated_right)

        # calculate gradients of probabilities with respect to examples
    gradients_up = autograd.grad(outputs=prob_interpolated_up, inputs=interpolated_up,
                                  grad_outputs=torch.ones(
                                      prob_interpolated_up.size()).cuda(),
                                  create_graph=True, retain_graph=True)[0]
    gradients_left = autograd.grad(outputs=prob_interpolated_left, inputs=interpolated_left,
                                 grad_outputs=torch.ones(
                                     prob_interpolated_left.size()).cuda(),
                                 create_graph=True, retain_graph=True)[0]
    gradients_right = autograd.grad(outputs=prob_interpolated_right, inputs=interpolated_right,
                                 grad_outputs=torch.ones(
                                     prob_interpolated_right.size()).cuda(),
                                 create_graph=True, retain_graph=True)[0]

    grad_penalty_up = ((gradients_up.norm(2, dim=1) - 1) ** 2).mean() * 10
    grad_penalty_left = ((gradients_left.norm(2, dim=1) - 1) ** 2).mean() * 10
    grad_penalty_right = ((gradients_right.norm(2, dim=1) - 1) ** 2).mean() * 10
    return grad_penalty_up,grad_penalty_left,grad_penalty_right




def calculate_rmse(actual_values, predicted_values):
    # 计算预测误差
    errors = actual_values - predicted_values

    # 计算平方误差的平均值
    mse = np.mean(errors ** 2)

    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mse)

    return rmse

def validate(val_loader,model_up,model_left,model_right):
    model_up.eval()
    model_left.eval()
    model_right.eval()
    rmses = []
    with torch.no_grad():
        for i,data in enumerate(val_loader):
            up_oppose = data['up']['oppose']
            up_inp = data['up']['inp']
            up_gt = data['up']['gt']
            up_gt = up_gt.numpy()

            left_oppose = data['left']['oppose']
            left_inp = data['left']['inp']
            left_gt = data['left']['gt']
            left_gt = left_gt.numpy()

            right_oppose = data['right']['oppose']
            right_inp = data['right']['inp']
            right_gt = data['right']['gt']
            right_gt = right_gt.numpy()

            # oppose_up, inp_up, gt_up = oppose_up.cuda(), inp_up.cuda(), gt_up.cuda()
            # oppose_left, inp_left, gt_left = oppose_left.cuda(), inp_left.cuda(), gt_left.cuda()
            # oppose_right, inp_right, gt_right = oppose_right.cuda(), inp_right.cuda(), gt_right.cuda()
            up_oppose = up_oppose.cuda()
            up_inp = up_inp.cuda()
            left_oppose = left_oppose.cuda()
            left_inp = left_inp.cuda()
            right_oppose = right_oppose.cuda()
            right_inp = right_inp.cuda()

            up = torch.cat([up_oppose, up_inp], 1)
            left = torch.cat([left_oppose, left_inp], 1)
            right = torch.cat([right_oppose, right_inp], 1)

            up_G_result = model_up(up)
            up_G_result = up_G_result.squeeze()
            gene_up = up_G_result.detach().cpu().numpy()

            left_G_result = model_left(left)
            left_G_result = left_G_result.squeeze()
            gene_left = left_G_result.detach().cpu().numpy()

            right_G_result = model_right(right)
            right_G_result = right_G_result.squeeze()
            gene_right = right_G_result.detach().cpu().numpy()

            now_rmse_up = calculate_rmse(up_gt, gene_up)
            now_rmse_left = calculate_rmse(left_gt,gene_left)
            now_rmse_right = calculate_rmse(right_gt,gene_right)

            now_rmse = 0.5*now_rmse_up+0.25*now_rmse_left+0.25*now_rmse_right

            rmses.append(now_rmse)
        rmses_mean = sum(rmses)/len(rmses)
    return rmses_mean

def validate_sig(val_loader,G):
    G.eval()

    rmses = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):

            up_inp = data['inp']
            up_gt = data['gt']
            gt = gt.numpy()
            inp = inp.cuda()
            G_result = G(inp)
            G_result = G_result.squeeze()
            gene_up = G_result.detach().cpu().numpy()
            now_rmse = calculate_rmse(gt, gene_up)

            rmses.append(now_rmse)
        rmses_mean = sum(rmses) / len(rmses)
    return rmses_mean




def validate_multi(val_loader,generator):
    generator.eval()
    rmses = []
    with torch.no_grad():
        for i,data in enumerate(val_loader):
            up_oppose = data['up']['oppose']
            up_inp = data['up']['inp']
            up_gt = data['up']['gt']
            up_mask = data['up']['mask']
            up_gt = up_gt.numpy()

            left_oppose = data['left']['oppose']
            left_inp = data['left']['inp']
            left_mask = data['left']['mask']
            left_gt = data['left']['gt']
            left_gt = left_gt.numpy()

            right_oppose = data['right']['oppose']
            right_inp = data['right']['inp']
            right_mask = data['right']['mask']
            right_gt = data['right']['gt']
            right_gt = right_gt.numpy()

            # oppose_up, inp_up, gt_up = oppose_up.cuda(), inp_up.cuda(), gt_up.cuda()
            # oppose_left, inp_left, gt_left = oppose_left.cuda(), inp_left.cuda(), gt_left.cuda()
            # oppose_right, inp_right, gt_right = oppose_right.cuda(), inp_right.cuda(), gt_right.cuda()
            up_oppose = up_oppose.cuda()
            up_inp = up_inp.cuda()
            up_mask = up_mask.cuda()
            left_oppose = left_oppose.cuda()
            left_inp = left_inp.cuda()
            left_mask = left_mask.cuda()
            right_oppose = right_oppose.cuda()
            right_inp = right_inp.cuda()
            right_mask = right_mask.cuda()

            up = torch.cat([up_oppose, up_inp], 1)
            left = torch.cat([left_oppose, left_inp], 1)
            right = torch.cat([right_oppose, right_inp], 1)

            up_G_result,left_G_result,right_G_result = generator(up,left,right,up_mask,left_mask,right_mask)
            up_G_result = up_G_result.squeeze()
            gene_up = up_G_result.detach().cpu().numpy()

            left_G_result = left_G_result.squeeze()
            gene_left = left_G_result.detach().cpu().numpy()

            right_G_result = right_G_result.squeeze()
            gene_right = right_G_result.detach().cpu().numpy()

            now_rmse_up = calculate_rmse(up_gt, gene_up)
            now_rmse_left = calculate_rmse(left_gt,gene_left)
            now_rmse_right = calculate_rmse(right_gt,gene_right)

            now_rmse = 0.5*now_rmse_up+0.25*now_rmse_left+0.25*now_rmse_right

            rmses.append(now_rmse)
        rmses_mean = sum(rmses)/len(rmses)
    return rmses_mean



# class ConstrastiveLoss(nn.Module):
#     def __init__(self, batch_size, temperature):
#         super(ConstrastiveLoss, self).__init__()
#         self.batch_size = batch_size
#         self.temperature = temperature
#
#         self.mask = self.mask_correlated_samples(batch_size)
#         self.criterion = nn.CrossEntropyLoss(reduction="sum")
#
#     #相关性矩阵
#     def mask_correlated_samples(self, batch_size):
#         N = 2 * batch_size
#         mask = torch.ones((N, N))
#         mask = mask.fill_diagonal_(0)
#         for i in range(batch_size):
#             mask[i, batch_size + i] = 0
#             mask[batch_size + i, i] = 0
#         mask = mask.bool()
#         return mask
#
#     def forward(self, z_i, z_j):
#         N = 2 * self.batch_size
#         z = torch.cat((z_i, z_j), dim=0)
#
#         #print(z.shape)
#         sim = torch.matmul(z, z.T) / self.temperature
#         sim_i_j = torch.diag(sim, self.batch_size)
#         sim_j_i = torch.diag(sim, self.batch_size)
#
#         positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
#         negative_samples = sim[self.mask].reshape(N, -1)
#
#         labels = torch.zeros(N).to(positive_samples.device).long()
#         logits = torch.cat((positive_samples, negative_samples), dim=1)
#         loss = self.criterion(logits, labels)
#         loss /= N
#         return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


class PartialConvolutionLoss(nn.Module):
    def __init__(self):
        super(PartialConvolutionLoss, self).__init__()

    def forward(self, x, mask, output, target,lamda_valid,lamda_hole):
        # L_valid: valid region loss
        L_valid = lamda_valid*torch.sum(torch.abs(output * mask - target * mask)) / torch.sum(mask)

        # L_hole: hole region loss
        L_hole = lamda_hole*torch.sum(torch.abs(output * (1 - mask))) / torch.sum(1 - mask)

        # Total loss
        total_loss = L_valid + L_hole

        return total_loss


def gram_matrix(input_tensor):
    """
    Compute Gram matrix

    :param input_tensor: input tensor with shape
     (batch_size, nbr_channels, height, width)
    :return: Gram matrix of y
    """
    (b, ch, h, w) = input_tensor.size()
    features = input_tensor.view(b, ch, w * h)
    features_t = features.transpose(1, 2)

    # more efficient and formal way to avoid underflow for mixed precision training
    input = torch.zeros(b, ch, ch).type(features.type())
    gram = torch.baddbmm(input, features, features_t, beta=0, alpha=1. / (ch * h * w), out=None)

    # naive way to avoid underflow for mixed precision training
    # features = features / (ch * h)
    # gram = features.bmm(features_t) / w

    # for fp32 training, it is also safe to use the following:
    # gram = features.bmm(features_t) / (ch * h * w)

    return gram

#
# class PerceptualLoss(nn.Module):
#     """
#     Perceptual Loss Module
#     """
#
#     def __init__(self):
#         """Init"""
#         super().__init__()
#         self.l1_loss = torch.nn.L1Loss()
#         self.mse_loss = torch.nn.MSELoss()
#
#     @staticmethod
#     def normalize_batch(batch, div_factor=255.):
#         """
#         Normalize batch
#
#         :param batch: input tensor with shape
#          (batch_size, nbr_channels, height, width)
#         :param div_factor: normalizing factor before data whitening
#         :return: normalized data, tensor with shape
#          (batch_size, nbr_channels, height, width)
#         """
#         # normalize using imagenet mean and std
#         mean = batch.data.new(batch.data.size())
#         std = batch.data.new(batch.data.size())
#         mean[:, 0, :, :] = 0.485
#         mean[:, 1, :, :] = 0.456
#         mean[:, 2, :, :] = 0.406
#         std[:, 0, :, :] = 0.229
#         std[:, 1, :, :] = 0.224
#         std[:, 2, :, :] = 0.225
#         batch = torch.div(batch, div_factor)
#
#         batch -= Variable(mean)
#         batch = torch.div(batch, Variable(std))
#         return batch
#
#     def forward(self, x, y):
#         """
#         Forward
#
#         :param x: input tensor with shape
#          (batch_size, nbr_channels, height, width)
#         :param y: input tensor with shape
#          (batch_size, nbr_channels, height, width)
#         :return: l1 loss between the normalized data
#         """
#         x = self.normalize_batch(x)
#         y = self.normalize_batch(y)
#         return self.l1_loss(x, y)
#
#
# def make_vgg16_layers(style_avg_pool=False):
#     """
#     make_vgg16_layers
#
#     Return a custom vgg16 feature module with avg pooling
#     """
#     vgg16_cfg = [
#         64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
#         512, 512, 512, 'M', 512, 512, 512, 'M'
#     ]
#
#     layers = []
#     in_channels = 3
#     for v in vgg16_cfg:
#         if v == 'M':
#             if style_avg_pool:
#                 layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)
#
#
# class VGG16Partial(nn.Module):
#     """
#     VGG16 partial model
#     """
#
#     def __init__(self, vgg_path='~/.torch/vgg16-397923af.pth', layer_num=3):
#         """
#         Init
#
#         :param layer_num: number of layers
#         """
#         super().__init__()
#         vgg_model = models.vgg16()
#         vgg_model.features = make_vgg16_layers()
#         vgg_model.load_state_dict(
#             torch.load(vgg_path, map_location='cpu')
#         )
#         vgg_pretrained_features = vgg_model.features
#
#         assert layer_num > 0
#         assert isinstance(layer_num, int)
#         self.layer_num = layer_num
#
#         self.slice1 = torch.nn.Sequential()
#         for x in range(5):  # 4
#             self.slice1.add_module(str(x), vgg_pretrained_features[x])
#
#         if self.layer_num > 1:
#             self.slice2 = torch.nn.Sequential()
#             for x in range(5, 10):  # (4, 9)
#                 self.slice2.add_module(str(x), vgg_pretrained_features[x])
#
#         if self.layer_num > 2:
#             self.slice3 = torch.nn.Sequential()
#             for x in range(10, 17):  # (9, 16)
#                 self.slice3.add_module(str(x), vgg_pretrained_features[x])
#
#         if self.layer_num > 3:
#             self.slice4 = torch.nn.Sequential()
#             for x in range(17, 24):  # (16, 23)
#                 self.slice4.add_module(str(x), vgg_pretrained_features[x])
#
#         for param in self.parameters():
#             param.requires_grad = False
#
#     @staticmethod
#     def normalize_batch(batch, div_factor=1.0):
#         """
#         Normalize batch
#
#         :param batch: input tensor with shape
#          (batch_size, nbr_channels, height, width)
#         :param div_factor: normalizing factor before data whitening
#         :return: normalized data, tensor with shape
#          (batch_size, nbr_channels, height, width)
#         """
#         # normalize using imagenet mean and std
#         mean = batch.data.new(batch.data.size())
#         std = batch.data.new(batch.data.size())
#         mean[:, 0, :, :] = 0.485
#         mean[:, 1, :, :] = 0.456
#         mean[:, 2, :, :] = 0.406
#         std[:, 0, :, :] = 0.229
#         std[:, 1, :, :] = 0.224
#         std[:, 2, :, :] = 0.225
#         batch = torch.div(batch, div_factor)
#
#         batch -= Variable(mean)
#         batch = torch.div(batch, Variable(std))
#         return batch
#
#     def forward(self, x):
#         """
#         Forward, get features used for perceptual loss
#
#         :param x: input tensor with shape
#          (batch_size, nbr_channels, height, width)
#         :return: list of self.layer_num feature maps used to compute the
#          perceptual loss
#         """
#         h = self.slice1(x)
#         h1 = h
#
#         output = []
#
#         if self.layer_num == 1:
#             output = [h1]
#         elif self.layer_num == 2:
#             h = self.slice2(h)
#             h2 = h
#             output = [h1, h2]
#         elif self.layer_num == 3:
#             h = self.slice2(h)
#             h2 = h
#             h = self.slice3(h)
#             h3 = h
#             output = [h1, h2, h3]
#         elif self.layer_num >= 4:
#             h = self.slice2(h)
#             h2 = h
#             h = self.slice3(h)
#             h3 = h
#             h = self.slice4(h)
#             h4 = h
#             output = [h1, h2, h3, h4]
#         return output
#
#
# class VGG16PartialLoss(PerceptualLoss):
#     """
#     VGG16 perceptual loss
#     """
#
#     def __init__(self, l1_alpha=5.0, perceptual_alpha=0.05, style_alpha=120,
#                  smooth_alpha=0, feat_num=3, vgg_path='~/.torch/vgg16-397923af.pth'):
#         """
#         Init
#
#         :param l1_alpha: weight of the l1 loss
#         :param perceptual_alpha: weight of the perceptual loss
#         :param style_alpha: weight of the style loss
#         :param smooth_alpha: weight of the regularizer
#         :param feat_num: number of feature maps
#         """
#         super().__init__()
#
#         self.vgg16partial = VGG16Partial(vgg_path=vgg_path).eval()
#
#         self.loss_fn = torch.nn.L1Loss(size_average=True)
#
#         self.l1_weight = l1_alpha
#         self.vgg_weight = perceptual_alpha
#         self.style_weight = style_alpha
#         self.regularize_weight = smooth_alpha
#
#         self.dividor = 1
#         self.feat_num = feat_num
#
#     def forward(self, output0, target0):
#         """
#         Forward
#
#         assuming both output0 and target0 are in the range of [0, 1]
#
#         :param output0: output of a model, tensor with shape
#          (batch_size, nbr_channels, height, width)
#         :param target0: target, tensor with shape
#          (batch_size, nbr_channels, height, width)
#         :return: total perceptual loss
#         """
#         y = self.normalize_batch(target0, self.dividor)
#         x = self.normalize_batch(output0, self.dividor)
#
#         # L1 loss
#         l1_loss = self.l1_weight * (torch.abs(x - y).mean())
#         vgg_loss = 0
#         style_loss = 0
#         smooth_loss = 0
#
#         # VGG
#         if self.vgg_weight != 0 or self.style_weight != 0:
#
#             yc = Variable(y.data)
#
#             with torch.no_grad():
#                 groundtruth = self.vgg16partial(yc)
#             generated = self.vgg16partial(x)
#
#             # vgg loss: VGG content loss
#             if self.vgg_weight > 0:
#                 # for m in range(0, len(generated)):
#                 for m in range(len(generated) - self.feat_num, len(generated)):
#                     gt_data = Variable(groundtruth[m].data, requires_grad=False)
#                     vgg_loss += (
#                             self.vgg_weight * self.loss_fn(generated[m], gt_data)
#                     )
#
#             # style loss: Gram matrix loss
#             if self.style_weight > 0:
#                 # for m in range(0, len(generated)):
#                 for m in range(len(generated) - self.feat_num, len(generated)):
#                     gt_style = gram_matrix(
#                         Variable(groundtruth[m].data, requires_grad=False))
#                     gen_style = gram_matrix(generated[m])
#                     style_loss += (
#                             self.style_weight * self.loss_fn(gen_style, gt_style)
#                     )
#
#         # smooth term
#         if self.regularize_weight != 0:
#             smooth_loss += self.regularize_weight * (
#                     torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).mean() +
#                     torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).mean()
#             )
#
#         tot = l1_loss + vgg_loss + style_loss + smooth_loss
#         return tot, vgg_loss, style_loss