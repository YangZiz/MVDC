from model import *
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from scipy.ndimage import label, find_objects

checkpoint = r'/yy/code/pix2pix/output/checkpoint/seg_right/best_segement.pth'
model = U_Net()
model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
test_dir = r'/yy/data/seg_right/test/inp'
save_dir = r'/yy/data/seg_right/test/output'
# def remove_small_areas(binary_array, min_area_threshold):
#     # 使用 label 函数进行连通域分析
#     labeled_array, num_features = label(binary_array)
#
#     # 找到各个连通域的区域
#     regions = find_objects(labeled_array)
#
#     # 遍历连通域
#     for region in regions:
#         # 计算连通域的面积
#         area = np.sum(binary_array[region] == 1)
#
#         # 如果面积小于阈值，则将该连通域置零
#         if area < min_area_threshold:
#             binary_array[region] = 0
#
#     return binary_array
def remove_small_components(matrix, threshold):
    labeled_matrix, num_labels = label(matrix)
    component_sizes = np.bincount(labeled_matrix.flatten())
    small_components = np.where(component_sizes < threshold)[0]
    for component in small_components:
        matrix[labeled_matrix == component] = 0
    return matrix
for name in os.listdir(test_dir):
    path = os.path.join(test_dir,name)
    inp = Image.open(path).convert('L')
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    output = model(inp).squeeze().cpu().detach().numpy()
    output[output>0.5] = 1
    output[output != 1] = 0
    esult_array = remove_small_components(output, 1000)
    esult_array[esult_array!=0] = 255
    #print(esult_array.max)

    image = Image.fromarray(esult_array.astype(np.uint8), 'L')
    image.save(os.path.join(save_dir, name))


