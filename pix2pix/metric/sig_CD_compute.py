import os

import numpy as np
from scipy.spatial import KDTree
import trimesh
import torch

def farthest_point_sample(xyz,npoint):
    '''
    最远点采样,返回采样点的索引
    xyz: Batch*N*(x,y,z),Tensor
    npoint: 采样点个数,int
    '''
    B,N,C=xyz.shape     # Batch,N,3
    # 1*512,用于记录选取的512个点的索引
    centroids=torch.zeros(B,npoint,dtype=torch.long)
    # 1*1024,用于记录1024个全部数据点与已采样点的距离
    distance=torch.ones(B,N)*1e10
    # [6],第一个点从0~N中随机选取,如点6;
    farthest=torch.randint(0,N,(B,),dtype=torch.long)
    # 一批点云数据的标号,[0]
    batch_indices=torch.arange(B,dtype=torch.long)
    for i in range(npoint):
        # 第一个点随机选取
        centroids[:,i]=farthest
        # 获取当前采样点的坐标,(x,y,z)
        centroid=xyz[batch_indices,farthest,:].view(B,1,3)
        # 计算1024个全部采样点与当前采样点的欧式距离
        dist=torch.sum((xyz-centroid)**2,-1)
        # 为更新每个点到已采样点的距离做标记
        mask=dist<distance
        # 更新每个点到已采样点的距离
        distance[mask]=dist[mask].float()
        # 选取到已采样点距离最大的点作为下一个采样点
        farthest=torch.max(distance,-1)[1]
    return centroids

def farthest_point_sampling(points, num_points):
    """
    最远点采样算法

    参数:
        points: numpy 数组，形状为 (N, D)，表示点云中的点，N 是点的数量，D 是每个点的维度
        num_points: 要采样的点的数量

    返回:
        sampled_points: numpy 数组，形状为 (num_points, D)，表示采样得到的点
    """
    # 随机选择一个起始点
    sampled_indices = [np.random.randint(0, len(points))]
    sampled_points = [points[sampled_indices[0]]]

    # 计算每个点到已选择点集合的最小距离
    for _ in range(1, num_points):
        distances = np.min(np.linalg.norm(points - sampled_points[-1], axis=1), axis=0)
        # 选择距离最远的点作为下一个采样点
        farthest_index = np.argmax(distances)
        sampled_indices.append(farthest_index)
        sampled_points.append(points[farthest_index])

    return np.array(sampled_points)
def l1_chamfer_distance(point_cloud1, point_cloud2):
    # 构建 KD 树
    kdtree1 = KDTree(point_cloud1)
    kdtree2 = KDTree(point_cloud2)

    # 计算每个点到另一个点云中的距离
    distances1, _ = kdtree1.query(point_cloud2)
    distances2, _ = kdtree2.query(point_cloud1)

    # 计算倒角距离（L1距离）
    chamfer_dist = np.mean(distances1) + np.mean(distances2)

    return chamfer_dist

def l2_chamfer_distance(point_cloud1, point_cloud2):
    # 构建 KD 树
    kdtree1 = KDTree(point_cloud1)
    kdtree2 = KDTree(point_cloud2)

    # 计算每个点到另一个点云中的距离
    distances1, _ = kdtree1.query(point_cloud2)
    distances2, _ = kdtree2.query(point_cloud1)

    # 计算倒角距离（L2距离）
    chamfer_dist = np.sqrt(np.mean(distances1**2) + np.mean(distances2**2))

    return chamfer_dist

test_root = r'C:\Users\Henry Su\Desktop\11_22test\pix2pix_pcd'


L1 = []
L2 = []
for id in os.listdir(test_root):
    #print(os.path.join(test_root,'\output\{}surface_reconstruction.off'.format(str(id))))
    #output_mesh = trimesh.load(r'E:\research\code\sa-ifn\root_reconstruction\yaguan\output\{}surface_reconstruction.off'.format(str(id)))
    output_mesh = trimesh.load(os.path.join(test_root,id))
    gt_mesh = trimesh.load(r'E:\teeth_data\multi_view\scaled_off\2_object\{}.off'.format(str(id[13:-4])))
    output_points = np.asarray(output_mesh.vertices)
    gt_points = np.asarray(gt_mesh.vertices)
    # sample_output_points = farthest_point_sampling(output_points,8192)
    # sample_gt_points = farthest_point_sampling(gt_points,8192)

    batches = torch.tensor([output_points])
    idx = farthest_point_sample(batches, 8192)
    downsampled_points = batches[0][idx]
    downsampled = downsampled_points.view(8192, 3)
    sample_output_points = np.asarray(downsampled)

    batches = torch.tensor([gt_points])
    idx = farthest_point_sample(batches, 8192)
    downsampled_points = batches[0][idx]
    downsampled = downsampled_points.view(8192, 3)
    sample_gt_points = np.asarray(downsampled)

    # print(sample_output_points.shape)
    # print(sample_gt_points.shape)
    print(id)
    l1_chamfer_dist = l1_chamfer_distance(sample_output_points,sample_gt_points)
    l2_chamfer_dist = l2_chamfer_distance(sample_output_points, sample_gt_points)
    L1.append(l1_chamfer_dist)
    L2.append(l2_chamfer_dist)

l1_mean = sum(L1)/len(L1)
l2_mean = sum(L2)/len(L2)
print(l1_mean,l2_mean)

