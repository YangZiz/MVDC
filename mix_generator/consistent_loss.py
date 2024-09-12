import numpy as np
import torch
import torch.nn as nn

class Consistent_loss_up(nn.Module):
    def __init__(self):
        super(Consistent_loss_up, self).__init__()

    def forward(self,up,left,right):
        threshold = 0.2
        up_point = []

        up2left = torch.zeros_like(up)
        up2right = torch.zeros_like(up)
        for k in range(up.shape[0]):
            up_point_k = []
            for i in range(up.shape[3]):
                for j in range(up.shape[2]):

                    if up[k][0][j][i]<0.0235:
                        continue
                    up_point_k.append([j,i,up[k,0,j,i]*50+110])
            up_point.append(up_point_k)

        for k in range(up.shape[0]):
            left_k = torch.zeros_like(up[0])
            right_k = torch.zeros_like(up[0])
            for i in range(len(up_point[k])):
                #left大于128的不考虑
                if up_point[k][i][0]>128:
                    if right_k[0][up_point[k][i][1]][torch.round(up_point[k][i][2]).to(torch.long)] < (up_point[k][i][0]-128) / 60:
                        right_k[0][up_point[k][i][1]][torch.round(up_point[k][i][2]).to(torch.long)] = (up_point[k][i][0]-128) / 60
                else:
                    if left_k[0][up_point[k][i][1]][torch.round(up_point[k][i][2]).to(torch.long)]<(128-up_point[k][i][0])/60:
                        left_k[0][up_point[k][i][1]][torch.round(up_point[k][i][2]).to(torch.long)] = (128-up_point[k][i][0])/60

            up2left[k] = left_k
            up2right[k] = right_k

        pixel_diff_u2l = torch.abs(up2left - left)

            # 应用阈值
        masked_diff_u2l = torch.where(pixel_diff_u2l < threshold, pixel_diff_u2l, torch.zeros_like(pixel_diff_u2l))

            # 计算 L1 损失
        l1_loss_u2l= torch.mean(masked_diff_u2l)

        pixel_diff_u2r = torch.abs(up2right - right)

            # 应用阈值
        masked_diff_u2r = torch.where(pixel_diff_u2r < threshold, pixel_diff_u2r, torch.zeros_like(pixel_diff_u2r))

            # 计算 L1 损失
        l1_loss_u2r = torch.mean(masked_diff_u2r)

        loss = l1_loss_u2r+l1_loss_u2l
        return loss


class Consistent_loss_up_2(nn.Module):
    def __init__(self):
        super(Consistent_loss_up_2, self).__init__()

    def forward(self,up,left,right):
        threshold = 0.2

        mask = (up>=0.0235)
        up_point = torch.nonzero(mask, as_tuple=True)
        up2left = torch.zeros_like(up)
        up2right = torch.zeros_like(up)
        #print(len(up_point[0]))
        for t in range(len(up_point[0])):
            k,i,j = up_point[0][t],up_point[2][t],up_point[3][t]
            if i >128:
                if up2right[k,0,j,torch.round(up[k,0,i,j]*50+110).to(torch.long)]<(i-128)/60:
                    up2right[k, 0, j, torch.round(up[k, 0, i, j] * 50 + 110).to(torch.long)] = (i - 128) / 60
            else:
                if up2left[k,0,j,torch.round(up[k,0,i,j]*50+110).to(torch.long)]<(128-i)/60:
                    up2left[k, 0, j, torch.round(up[k, 0, i, j] * 50 + 110).to(torch.long)] = (128-i) / 60

        pixel_diff_u2l = torch.abs(up2left - left)

            # 应用阈值
        masked_diff_u2l = torch.where((pixel_diff_u2l < threshold) & (up2left!=0), pixel_diff_u2l, torch.zeros_like(pixel_diff_u2l))

            # 计算 L1 损失
        l1_loss_u2l= torch.mean(masked_diff_u2l)

        pixel_diff_u2r = torch.abs(up2right - right)

            # 应用阈值
        masked_diff_u2r = torch.where((pixel_diff_u2r < threshold) & (up2right!=0), pixel_diff_u2r, torch.zeros_like(pixel_diff_u2r))

            # 计算 L1 损失
        l1_loss_u2r = torch.mean(masked_diff_u2r)

        loss = l1_loss_u2r+l1_loss_u2l
        return loss


class Consistent_loss_up_3(nn.Module):
    def __init__(self):
        super(Consistent_loss_up_3, self).__init__()

    def forward(self, up_output, left_output, right_output):
        threshold = 0.2
        #This batch_size=1 test
        up = up_output.clone().squeeze()
        left = left_output.clone().squeeze()
        right = right_output.clone().squeeze()
        mask = (up >= 0.0235)
        up_point = torch.nonzero(mask, as_tuple=True)
        up2left = torch.zeros_like(up)
        up2right = torch.zeros_like(up)
        # print(len(up_point[0]))
        for t in range(len(up_point[0])):
            i, j = up_point[0][t], up_point[1][t]
            if i > 128:
                if up2right[j, torch.round(up[i, j] * 50 + 110).to(torch.long)] < (i - 128) / 60:
                    up2right[j, torch.round(up[i, j] * 50 + 110).to(torch.long)] = (i - 128) / 60
            else:
                if up2left[j, torch.round(up[i, j] * 50 + 110).to(torch.long)] < (128 - i) / 60:
                    up2left[j, torch.round(up[i, j] * 50 + 110).to(torch.long)] = (128 - i) / 60

        pixel_diff_u2l = torch.abs(up2left - left)

        # 应用阈值
        masked_diff_u2l = torch.where((pixel_diff_u2l < threshold) & (up2left != 0), pixel_diff_u2l,
                                      torch.zeros_like(pixel_diff_u2l))

        # 计算 L1 损失
        l1_loss_u2l = torch.mean(masked_diff_u2l)

        pixel_diff_u2r = torch.abs(up2right - right)

        # 应用阈值
        masked_diff_u2r = torch.where((pixel_diff_u2r < threshold) & (up2right != 0), pixel_diff_u2r,torch.zeros_like(pixel_diff_u2r))

        # 计算 L1 损失
        l1_loss_u2r = torch.mean(masked_diff_u2r)

        loss = l1_loss_u2r + l1_loss_u2l
        return loss


class Consistent_loss_up_4(nn.Module):
    def __init__(self):
        super(Consistent_loss_up_4, self).__init__()

    def forward(self, up_output, left_output, right_output):
        batch_size = up_output.size(0)
        up = up_output.clone().squeeze()
        left = left_output.clone().squeeze()
        right = right_output.clone().squeeze()

        u = torch.arange(0,256)
        v = torch.arange(0,256)
        u,v = torch.meshgrid(u,v)
        u = u.to(dtype=torch.float)
        v = v.to(dtype=torch.float)
        Z = up*50+110

        X = u.view(-1).cuda()
        Y = v.view(-1).cuda()
        Z = Z.view(batch_size,-1)
        Z = torch.round(Z)
        loss = 0
        for i in range(batch_size):
            valid_left = X<128
            valid_right = X>=128

            X_left = X[valid_left]
            Y_left = Y[valid_left]
            Z_left = Z[i,valid_left]
            X_right = X[valid_right ]
            Y_right  = Y[valid_right ]
            Z_right  = Z[i,valid_right ]

            #将_left投影为up2left,_right投影为up2right
            X_left = (128-X_left)/60
            X_right = (X_right-128)/60
            up2left = torch.zeros((256,256),dtype=up.dtype).cuda()
            up2right = torch.zeros((256, 256), dtype=up.dtype).cuda()

            sorted_indices_left = torch.argsort(X_left, descending=True)
            sorted_Y_left = Y_left[sorted_indices_left]
            sorted_Z_left = Z_left[sorted_indices_left]
            sorted_X_left = X_left[sorted_indices_left]
            indices_left = sorted_Y_left * 256 + sorted_Z_left
            indices_left_copy = indices_left.clone()
            indices_left_np = indices_left_copy.cpu().detach().numpy()
            _, ind = np.unique(indices_left_np, return_index=True)
            remove_indices_left = indices_left[ind]
            remove_X_left = sorted_X_left[ind]
        #up2left.view(-1)[remove_indices_left] = remove_X_left
        #print(remove_indices_left.type)
            up2left.view(-1).scatter_(0,remove_indices_left.to(torch.int64),remove_X_left)

        # 将深度图还原成二维数组
            up2left = up2left.view(256, 256)

            sorted_indices_right = torch.argsort(X_right, descending=True)
            sorted_Y_right = Y_right[sorted_indices_right]
            sorted_Z_right = Z_right[sorted_indices_right]
            sorted_X_right = X_right[sorted_indices_right]
            indices_right = sorted_Y_right * 256 + sorted_Z_right

            indices_right_copy = indices_right.clone()
            indices_right_np = indices_right_copy.cpu().detach().numpy()
            _, ind = np.unique(indices_right_np, return_index=True)
            remove_indices_right = indices_right[ind]
            remove_X_right = sorted_X_right[ind]
            up2right.view(-1).scatter_(0,remove_indices_right.to(torch.int64),remove_X_right)

            up2right = up2right.view(256,256)



            pixel_diff_u2l = torch.abs(up2left - left[i])

        # 应用阈值
            masked_diff_u2l = torch.where((pixel_diff_u2l < 0.2) & (up2left != 0), pixel_diff_u2l,
                                      torch.zeros_like(pixel_diff_u2l))

        # 计算 L1 损失
            l1_loss_u2l = torch.mean(masked_diff_u2l)

            pixel_diff_u2r = torch.abs(up2right - right[i])

        # 应用阈值
            masked_diff_u2r = torch.where((pixel_diff_u2r < 0.2) & (up2right != 0), pixel_diff_u2r,torch.zeros_like(pixel_diff_u2r))

        # 计算 L1 损失
            l1_loss_u2r = torch.mean(masked_diff_u2r)

            loss += l1_loss_u2r + l1_loss_u2l
        loss /= batch_size
        return loss

class Consistent_loss_left(nn.Module):
    def __init__(self):
        super(Consistent_loss_left, self).__init__()

    def forward(self, up, left, right):
        threshold = 0.2
        left_point = []

        left2up = torch.zeros_like(up)
        for k in range(up.shape[0]):
            left_point_k = []
            for i in range(up.shape[3]):
                for j in range(up.shape[2]):

                    if left[k][0][j][i] < 0.0235:
                        continue
                    left_point_k.append([128-left[k][0][j][i]*60,j,i])
            left_point.append(left_point_k)

        for k in range(up.shape[0]):
            up_k = torch.zeros_like(up[0])
            for i in range(len(left_point[k])):
                        # left大于128的不考虑
                if left_point[k][i][2] <110:
                    if up_k[0][torch.round(left_point[k][i][0]).to(torch.long)][left_point[k][i][1]] < (110-left_point[k][i][2] ) / 50:
                        up_k[0][torch.round(left_point[k][i][0]).to(torch.long)][left_point[k][i][1]] = (110-left_point[k][i][2]) / 50

            left2up[k] = up_k


        pixel_diff_l2u = torch.abs(left2up - up)

                # 应用阈值
        masked_diff_l2u = torch.where(pixel_diff_l2u < threshold, pixel_diff_l2u,torch.zeros_like(pixel_diff_l2u))

                # 计算 L1 损失
        l1_loss_l2u = torch.mean(masked_diff_l2u)


        loss = l1_loss_l2u
        return loss

# class Consistent_loss_right(nn.Module):
#     def __init__(self):
#         super(Consistent_loss_right, self).__init__()
#
#     def forward(self, up, left, right):
#         threshold = 0.2
#         right_point = []
#
#         right2up = torch.zeros_like(up)
#         for k in range(up.shape[0]):
#             right_point_k = []
#             for i in range(up.shape[3]):
#                 for j in range(up.shape[2]):
#
#                     if right[k][0][j][i] < 0.0235:
#                         continue
#                     right_point_k.append([right[k][0][j][i] * 60+128, j, i])
#             right_point.append(right_point_k)
#
#         for k in range(up.shape[0]):
#             up_k = torch.zeros_like(up[0])
#             for i in range(len(right_point[k])):
#                         # left大于128的不考虑
#                 if right_point[k][i][2] < 110:
#                     if up_k[0][torch.round(right_point[k][i][0]).to(torch.long)][right_point[k][i][1]] > (110 - right_point[k][i][2]) / 50:
#                         up_k[0][torch.round(right_point[k][i][0]).to(torch.long)][right_point[k][i][1]] = (110 -right_point[k][i][2]) / 50
#
#             right2up[k] = up_k
#
#         pixel_diff_r2u = torch.abs(right2up - up)
#
#                 # 应用阈值
#         masked_diff_r2u = torch.where(pixel_diff_r2u < threshold, pixel_diff_r2u,torch.zeros_like(pixel_diff_r2u))
#
#                 # 计算 L1 损失
#         l1_loss_r2u = torch.mean(masked_diff_r2u)
#
#         loss = l1_loss_r2u
#         return loss

class Consistent_loss_right(nn.Module):
    def __init__(self):
        super(Consistent_loss_right, self).__init__()

    def forward(self, up, left, right):
        threshold = 0.2
        right_point = []

        right2up = torch.zeros_like(up)
        for k in range(up.shape[0]):
            right_point_k = []
            for i in range(up.shape[3]):
                for j in range(up.shape[2]):

                    if right[k][0][j][i] < 0.0235:
                        continue
                    right_point_k.append([right[k][0][j][i] * 60+128, j, i])
            right_point.append(right_point_k)

        mask = (right < 0.0235)
        right_point = torch.nonzero(mask, as_tuple=True)

        for t in range(len(right_point)):
            k,i,j = right_point[t][0],right_point[t][2],right_point[t][3]

        for k in range(up.shape[0]):
            up_k = torch.zeros_like(up[0])
            for i in range(len(right_point[k])):
                        # left大于128的不考虑
                if right_point[k][i][2] < 110:
                    if up_k[0][torch.round(right_point[k][i][0]).to(torch.long)][right_point[k][i][1]] > (110 - right_point[k][i][2]) / 50:
                        up_k[0][torch.round(right_point[k][i][0]).to(torch.long)][right_point[k][i][1]] = (110 -right_point[k][i][2]) / 50

            right2up[k] = up_k

        pixel_diff_r2u = torch.abs(right2up - up)

                # 应用阈值
        masked_diff_r2u = torch.where(pixel_diff_r2u < threshold, pixel_diff_r2u,torch.zeros_like(pixel_diff_r2u))

                # 计算 L1 损失
        l1_loss_r2u = torch.mean(masked_diff_r2u)

        loss = l1_loss_r2u
        return loss


