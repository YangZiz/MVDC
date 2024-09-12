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


warnings.filterwarnings('ignore')


# 计算特征提取模块的感知损失
def vgg16_loss(feature_module,loss_func,y,y_):
    out=feature_module(y)
    out_=feature_module(y_)
    loss=loss_func(out,out_)
    return loss

# 获取指定的特征提取模块
def get_feature_module(layer_index,device=None):
    vgg = vgg16(pretrained=True, progress=True).features
    vgg.cuda()
    vgg.eval()

    # 冻结参数
    for parm in vgg.parameters():
        parm.requires_grad = False

    feature_module = vgg[0:layer_index + 1]
    feature_module.to(device)
    return feature_module


# 计算指定的组合模块的感知损失
class PerceptualLoss(nn.Module):
    def __init__(self,loss_func,layer_indexs=None,device=None):
        super(PerceptualLoss, self).__init__()
        self.creation=loss_func
        self.layer_indexs=layer_indexs
        self.device=device

    def forward(self,y,y_):
        loss=0
        for index in self.layer_indexs:
            feature_module=get_feature_module(index,self.device)
            loss+=vgg16_loss(feature_module,self.creation,y,y_)
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
    for i in range(image_list.shape[0]):
        image = image_list[i]
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
        mask = mask * 255
        # print(mask)
        # mask = mask.repeat(3, axis=0)
        mask = mask.transpose(1, 2, 0)
        mask = np.squeeze(mask)
        mask_rio = image[center_x-len_s//2:center_x-len_s//2+len_s,center_y-len_s//2:center_y-len_s//2+len_s].clone()
        mask_renge = rio[center_x-len_s//2:center_x-len_s//2+len_s,center_y-len_s//2:center_y-len_s//2+len_s].clone()
        mask_renge = mask_renge.numpy()
        mask_rio[(mask == 0) & (mask_renge == 1)] = 0
        image[center_x-len_s//2:center_x-len_s//2+len_s,center_y-len_s//2:center_y-len_s//2+len_s] = mask_rio
        new_image[i] = image
    return new_image


def save_inter(inp,output,save_root,epoch):
    save_dir = os.path.join(save_root,'{}'.format(epoch))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(inp.shape[0]):
        inp_i = inp[i]
        oppose = inp_i[0].cpu().detach().numpy()
        partial = inp_i[1].cpu().detach().numpy()
        predict = output[i].squeeze().cpu().detach().numpy()

        inp_oppose = oppose * 255
        inp_oppose[inp_oppose < 0] = 0
        inp_oppose[inp_oppose > 255] = 255

        inp_partial = partial * 255
        inp_partial[inp_partial < 0] = 0
        inp_partial[inp_partial > 255] = 255

        gene_image = predict * 255
        gene_image[gene_image < 0] = 0
        gene_image[gene_image > 255] = 255

        image_oppose = Image.fromarray(inp_oppose.astype(np.uint8),'L')
        image_oppose.save(os.path.join(save_dir, '{}_oppose.png'.format(i+1)))

        image_partial = Image.fromarray(inp_partial.astype(np.uint8), 'L')
        image_partial.save(os.path.join(save_dir, '{}_partial.png'.format(i + 1)))

        image = Image.fromarray(gene_image.astype(np.uint8), 'L')
        image.save(os.path.join(save_dir, '{}_predict.png'.format(i+1)))



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


