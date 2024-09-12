import os

from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.feature import CENSURE
from scipy.stats import entropy
from skimage import color

def calculate_psnr(predicted_path, actual_path,inp_path):
    predicted_image = Image.open(predicted_path)
    actual_image = Image.open(actual_path)
    inp_image = Image.open(inp_path)

    # 将图像转换为NumPy数组
    predicted_array = np.array(predicted_image)
    actual_array = np.array(actual_image)
    inp_array = np.array(inp_image)
    mask = (inp_array == 0) & (actual_array != 0)

    mse = np.mean((predicted_array[mask] - actual_array[mask]) ** 2)

    # 计算峰值信噪比（PSNR）
    max_pixel = np.max(actual_array)
    psnr = 10 * np.log10(max_pixel ** 2 / mse)

    return psnr



def calculate_image_rmse(predicted_path, actual_path,inp_path):
    # 读取两个图像文件
    predicted_image = Image.open(predicted_path)
    actual_image = Image.open(actual_path)
    inp_image = Image.open(inp_path)

    # 将图像转换为NumPy数组
    predicted_array = np.array(predicted_image)
    actual_array = np.array(actual_image)
    inp_array = np.array(inp_image)
    mask = (inp_array == 0) & (actual_array != 0)

    # 计算差值的平方
    diff = (predicted_array[mask] - actual_array[mask]) ** 2
    mse = np.mean(diff)

    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mse)

    # 计算RMSE
    #rmse = np.sqrt(np.mean((predicted_array - actual_array) ** 2))

    return rmse

def calculate_msssim(predicted_path, actual_path):
    # 读取两个图像文件
    predicted_image = Image.open(predicted_path).convert('L')  # 转为灰度图
    actual_image = Image.open(actual_path).convert('L')  # 转为灰度图

    # 将图像转换为NumPy数组
    predicted_array = np.array(predicted_image)
    actual_array = np.array(actual_image)

    # 计算MS-SSIM
    msssim_index, _ = ssim(predicted_array, actual_array, full=True)

    return msssim_index

def calculate_fsim(predicted_path, actual_path):
    # 读取两个图像文件
    predicted_image = Image.open(predicted_path).convert('L')  # 转为灰度图
    actual_image = Image.open(actual_path).convert('L')  # 转为灰度图

    # 将图像转换为NumPy数组
    predicted_array = np.array(predicted_image)
    actual_array = np.array(actual_image)

    # 提取图像的关键特征点
    keypoints_predicted = CENSURE().detect(predicted_array)
    keypoints_actual = CENSURE().detect(actual_array)

    # 计算 FSIM 的特征相似性
    fsim_feature = ssim(predicted_array, actual_array)

    # 计算 FSIM 的局部相似性
    fsim_local = np.mean([ssim(predicted_array, actual_array, win_size=block_size, center=(int(keypoint[0]), int(keypoint[1])))
                          for keypoint in keypoints_predicted])

    # 计算 FSIM 的结构相似性
    fsim_structure = ssim(color.rgb2gray(predicted_array), color.rgb2gray(actual_array))

    # 计算 FSIM 的灰度直方图相似性
    hist_predicted, _ = np.histogram(predicted_array, bins=256, range=(0, 256), density=True)
    hist_actual, _ = np.histogram(actual_array, bins=256, range=(0, 256), density=True)
    fsim_histogram = np.sum(np.minimum(hist_predicted, hist_actual))

    # 计算 FSIM 的频谱相似性
    fft_predicted = np.fft.fft2(predicted_array)
    fft_actual = np.fft.fft2(actual_array)
    fsim_spectrum = np.sum(np.abs(np.log(np.abs(fft_predicted) + 1) - np.log(np.abs(fft_actual) + 1)))

    # 计算 FSIM 的最终值
    fsim_value = fsim_feature * fsim_local * fsim_structure * fsim_histogram * fsim_spectrum

    return fsim_value


gt_root_up = r'E:\teeth_data\multi_view\depth_map\up\2_object'
output_root = r'C:\Users\Henry Su\Desktop\11_22test\1_18_gro\test'
inp_root_up = r'E:\teeth_data\multi_view\depth_map\up\5_inp'
# all_rmses = []
# all_psnrs = []
rmses = []
psnrs = []
for id in os.listdir(output_root):
    gt_path = os.path.join(gt_root_up,'voxelization_data{}.png'.format(str(id)))
    output_path = os.path.join(output_root,id,'up{}.png'.format(str(id)))
    inp_path = os.path.join(inp_root_up,'voxelization_data{}.png'.format(str(id)))
    rmse = calculate_image_rmse(gt_path,output_path,inp_path)
    rmses.append(rmse)
    psnr = calculate_psnr(gt_path, output_path, inp_path)
    psnrs.append(psnr)
rmses_mean = sum(rmses)/len(rmses)
print(rmses_mean)

psnrs_mean = sum(psnrs)/len(psnrs)
print(psnrs_mean)
# all_rmses.append(rmses_mean)
# all_psnrs.append(psnrs_mean)

# #left
# gt_root_left = r'E:\teeth_data\multi_view\depth_map\left\2_object'
# output_root = r'C:\Users\Henry Su\Desktop\11_22test\1_16\test'
# inp_root_left = r'E:\teeth_data\multi_view\depth_map\left\5_inp'
# rmses = []
#
# psnrs = []
# for id in os.listdir(output_root):
#     gt_path = os.path.join(gt_root_left,'voxelization_data{}.png'.format(str(id)))
#     output_path = os.path.join(output_root,id,'left{}.png'.format(str(id)))
#     inp_path = os.path.join(inp_root_left,'voxelization_data{}.png'.format(str(id)))
#     rmse = calculate_image_rmse(gt_path,output_path,inp_path)
#     rmses.append(rmse)
#     psnr = calculate_psnr(gt_path, output_path, inp_path)
#     psnrs.append(psnr)
# rmses_mean = sum(rmses)/len(rmses)
# print(rmses_mean)
#
# psnrs_mean = sum(psnrs)/len(psnrs)
# print(psnrs_mean)
# all_rmses.append(rmses_mean)
# all_psnrs.append(psnrs_mean)
# #right
# gt_root_right= r'E:\teeth_data\multi_view\depth_map\right\2_object'
# output_root = r'C:\Users\Henry Su\Desktop\11_22test\1_16\test'
# inp_root_right = r'E:\teeth_data\multi_view\depth_map\right\5_inp'
# rmses = []
#
# psnrs = []
# for id in os.listdir(output_root):
#     gt_path = os.path.join(gt_root_right,'voxelization_data{}.png'.format(str(id)))
#     output_path = os.path.join(output_root,id,'right{}.png'.format(str(id)))
#     inp_path = os.path.join(inp_root_right,'voxelization_data{}.png'.format(str(id)))
#     rmse = calculate_image_rmse(gt_path,output_path,inp_path)
#     rmses.append(rmse)
#     psnr = calculate_psnr(gt_path, output_path, inp_path)
#     psnrs.append(psnr)
# rmses_mean = sum(rmses)/len(rmses)
# print(rmses_mean)
#
# psnrs_mean = sum(psnrs)/len(psnrs)
# print(psnrs_mean)
# all_rmses.append(rmses_mean)
# all_psnrs.append(psnrs_mean)
#
# print(sum(all_rmses)/len(all_rmses),sum(all_psnrs)/len(all_psnrs))