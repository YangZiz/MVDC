import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
from mutil_view_model import Generator
import torch


def calculate_rmse(actual_values, predicted_values):
    # 计算预测误差
    errors = actual_values - predicted_values

    # 计算平方误差的平均值
    mse = np.mean(errors ** 2)

    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mse)

    return rmse

def calculate_psnr(original_image, compressed_image):
    # 计算均方误差（MSE）
    mse = np.mean((original_image - compressed_image) ** 2)

    # 计算峰值信噪比（PSNR）
    max_pixel_value = 255  # 对于8位图像
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)

    return psnr


test_dir = r'/yy/data/depth_map'
save_dir = r'/yy/data/depth_map/11_27test'
checkpoint_dir = r'/yy/code/pix2pix/output/checkpoint/11_27'
filenames = os.listdir(os.path.join(test_dir,'up','6_test'))

up_G = Generator()
left_G = Generator()
right_G = Generator()
up_G.load_state_dict(torch.load(os.path.join(checkpoint_dir,'best_generator.pth'),map_location='cpu')['up_model'])
# left_G.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_generator.pth'))['left_model'])
# right_G.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_generator.pth'))['right_model'])


filenames = os.listdir(os.path.join(test_dir,'up','6_test'))
rmses = []
psnrs = []

for name in filenames:
    #up
    save_up_path = os.path.join(save_dir,'up',name)
    # save_left_path = os.path.join(save_dir,'left',name)
    # save_right_path = os.path.join(save_dir,'right',name)
    up_oppose = Image.open(os.path.join(test_dir,'up','1_oppose',name)).convert('L')
    up_oppose = up_oppose.rotate(270)
    up_inp = Image.open(os.path.join(test_dir,'up','6_test',name)).convert('L')
    up_inp = up_inp.rotate(270)
    up_target = Image.open(os.path.join(test_dir,'up','2_object',name)).convert('L')
    up_target = up_target.rotate(270)


    up_oppose = transforms.ToTensor()(up_oppose)
    up_inp = transforms.ToTensor()(up_inp)
    up_oppose = up_oppose.unsqueeze(0).cpu()
    up_inp = up_inp.unsqueeze(0).cpu()
    up_target = transforms.ToTensor()(up_target).cpu().squeeze()
    up_target = up_target.numpy()
    print(up_target.shape)


    up = torch.cat([up_oppose, up_inp], 1)


    up_G_result = up_G(up)



    up_G_result = up_G_result.squeeze()

    gene_up = up_G_result.detach().numpy()
    print(gene_up.shape)
    now_rmse = calculate_rmse(up_target, gene_up)
    now_psnr = calculate_psnr(up_target, gene_up)
    rmses.append(now_rmse)
    psnrs.append(now_psnr)

    gene_up = gene_up*255





    gene_up[gene_up<0] = 0

    gene_up[gene_up>255] = 255



    up_image = Image.fromarray(gene_up.astype(np.uint8),'L')


    up_image.save(save_up_path)

rmses_mean = sum(rmses)/len(rmses)
psnrs_mean = sum(psnrs)/len(psnrs)
print(rmses_mean,psnrs_mean)


