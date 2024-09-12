from consistent_loss import *
from PIL import Image
import numpy as np
import os




up_path = r'E:\teeth_data\multi_view\depth_map\up\2_object\voxelization_data0680.png'
left_path = r'E:\teeth_data\multi_view\depth_map\left\2_object\voxelization_data0680.png'
right_path = r'E:\teeth_data\multi_view\depth_map\right\2_object\voxelization_data0680.png'

# 读取灰度图像

up_image = Image.open(up_path).convert('L')
# 将图像转换为NumPy数组
up = np.array(up_image)

left_image = Image.open(left_path).convert('L')
left = np.array(left_image)

right_image = Image.open(right_path).convert('L')
right = np.array(right_image)

#归一化
up = up/255
left = left/255
right = right/255


up_point = []
for i in range(up.shape[1]):
    for j in range(up.shape[0]):
        if up[j][i]<0.0235:
            continue
        up_point.append([j,i,up[j][i]*50+110])
up2left = np.zeros(up.shape,dtype=up.dtype)
up2right = np.zeros(up.shape,dtype=up.dtype)

for i in range(len(up_point)):
    # left大于128的不考虑
    if up_point[i][0] > 128:
        if up2right[up_point[i][1]][round(up_point[i][2])] < (up_point[i][0] - 128) / 60:
            up2right[up_point[i][1]][round(up_point[i][2])] = (up_point[i][0] - 128) / 60
    else:
        if up2left[up_point[i][1]][round(up_point[i][2])] < (128 - up_point[i][0]) / 60:
            up2left[up_point[i][1]][round(up_point[i][2])] = (128 - up_point[i][0]) / 60

print(up2left.max())

up2left_raw = up2left*255
image_u2l = Image.fromarray(up2left_raw.astype(np.uint8),'L')
image_u2l.save(r'C:\Users\Henry Su\Desktop\u2l.png')


pixel_diff_u2l = np.abs(left-up2left)
masked_diff_u2l = pixel_diff_u2l.copy()
masked_diff_u2l[(up2left==0)] = 0
nonzero_indices = np.nonzero(masked_diff_u2l)

# 获取非零元素的值
nonzero_values = masked_diff_u2l[nonzero_indices]

# 计算非零元素的平均值
average_nonzero = np.mean(nonzero_values)
print(average_nonzero)


print(masked_diff_u2l.max())
print(masked_diff_u2l.min())


            # 计算 L1 损失
l1_loss_u2l= np.sum(masked_diff_u2l)




pixel_diff_u2r = np.abs(right-up2right)
masked_diff_u2r = pixel_diff_u2r.copy()
masked_diff_u2r[pixel_diff_u2r>0.2] = 0

            # 计算 L1 损失
l1_loss_u2r= np.sum(masked_diff_u2r)

print(l1_loss_u2l,l1_loss_u2r)



