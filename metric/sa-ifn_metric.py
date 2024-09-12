import numpy as np
from PIL import Image
import os

image_dir = r'E:\research\code\sa-ifn\root_reconstruction\yaguan\image_metric\scale\voxel\inp'
names_dir = r'E:\research\code\sa-ifn\root_reconstruction\yaguan\image_metric\scale\voxel\gt'
gt_dir = r'E:\research\code\sa-ifn\root_reconstruction\yaguan\image_metric\scale\voxel\gt'
save_dir = r'E:\research\code\sa-ifn\root_reconstruction\yaguan\image_metric\depth_image\inp'
filenames = os.listdir(names_dir)
for name in filenames:
    # if os.path.isfile(os.path.join(save_dir,name[:-4]+'png')):
    #     continue
    print(name)
    image_path = os.path.join(image_dir, name)
    gt_path = os.path.join(gt_dir, name)
    occ1 = np.unpackbits(np.load(image_path))
    inp = np.reshape(occ1, (256,) * 3)
    occ2 = np.unpackbits(np.load(gt_path))
    gt = np.reshape(occ2, (256,) * 3)

    nonzero_indices = np.nonzero(gt)

# 获取z坐标的最小值和最大值
    min_z = np.min(nonzero_indices[2])
    max_z = np.max(nonzero_indices[2])
    inp_map = np.zeros((256,256),dtype="float32")
    gt_map = np.zeros((256,256),dtype="float32")
    for i in range(256):
        for j in range(256):
            z_gt_indices = np.where(inp[i, j, :] != 0)[0]
            if len(z_gt_indices) == 0:
                gt_map[i, j] = 0
            else:
                gt_map[i, j] = (max(z_gt_indices)-130)*255/(max_z-130)
    non_zero_values = gt_map[gt_map != 0]
    gt_map[gt_map<0] = 0
    gt_map[gt_map>255] = 255
    image_gt = Image.fromarray(gt_map.astype(np.uint8),'L')
    image_gt.save(os.path.join(save_dir,name[:-4]+'.png'))

