import os
from PIL import Image
import torchvision.transforms as transforms
import torch
from mutil_view_model import Mutil_view_PGenerator
import numpy as np
from utils import *
def validate_multi(data,generator):
    up_oppose = data['up']['oppose']
    up_inp = data['up']['inp']
    #up_gt = data['up']['gt']
    up_mask = data['up']['mask']
    #up_gt = up_gt.numpy()

    left_oppose = data['left']['oppose']
    left_inp = data['left']['inp']
    left_mask = data['left']['mask']
    # left_gt = data['left']['gt']
    # left_gt = left_gt.numpy()

    right_oppose = data['right']['oppose']
    right_inp = data['right']['inp']
    right_mask = data['right']['mask']
    # right_gt = data['right']['gt']
    # right_gt = right_gt.numpy()

            # oppose_up, inp_up, gt_up = oppose_up.cuda(), inp_up.cuda(), gt_up.cuda()
            # oppose_left, inp_left, gt_left = oppose_left.cuda(), inp_left.cuda(), gt_left.cuda()
            # oppose_right, inp_right, gt_right = oppose_right.cuda(), inp_right.cuda(), gt_right.cuda()
    up_oppose = up_oppose.cpu()
    up_inp = up_inp.cpu()
    up_mask = up_mask.cpu()
    left_oppose = left_oppose.cpu()
    left_inp = left_inp.cpu()
    left_mask = left_mask.cpu()
    right_oppose = right_oppose.cpu()
    right_inp = right_inp.cpu()
    right_mask = right_mask.cpu()

    up = torch.cat([up_oppose, up_inp], 1)
    left = torch.cat([left_oppose, left_inp], 1)
    right = torch.cat([right_oppose, right_inp], 1)
    print(up.shape)

    up_G_result,left_G_result,right_G_result = generator(up,left,right,up_mask,left_mask,right_mask)
    up_G_result = up_G_result.squeeze()
    gene_up = up_G_result.detach().numpy()

    left_G_result = left_G_result.squeeze()
    gene_left = left_G_result.detach().numpy()

    right_G_result = right_G_result.squeeze()
    gene_right = right_G_result.detach().numpy()
    return gene_up,gene_left,gene_right



if __name__=="__main__":
    checkpoint_dir = r'/yy/code/pix2pix/output/checkpoint/1_14_pretrain_and_partial'
    generator = Mutil_view_PGenerator()
    generator.load_state_dict(torch.load(os.path.join(checkpoint_dir,'best_generator.pth'),map_location='cpu')['model'])
    test_dir = r'/yy/data/depth_map/multi_view'
    generator.eval()
    save_root = r'/yy/data/depth_map/1_28_test'
    files_names = os.listdir(r'/yy/data/depth_map/multi_view/up/2_object/test')
    for name in files_names:
        save_dir = os.path.join(save_root,'{}'.format(name[-8:-4]))
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        save_up_path = os.path.join(save_dir,'up'+name[-8:])
        save_left_path = os.path.join(save_dir, 'left' + name[-8:])
        save_right_path = os.path.join(save_dir, 'right' + name[-8:])
        save_up_mask_path = os.path.join(save_dir,'up_mask'+name[-8:])
        save_left_mask_path = os.path.join(save_dir, 'left_mask' + name[-8:])
        save_right_mask_path = os.path.join(save_dir, 'right_mask' + name[-8:])
        data = {}
        data['up'] = {}
        data['left'] = {}
        data['right'] = {}
        mask_add = torch.ones((1, 256, 256), dtype=torch.float32)



        oppose = Image.open(os.path.join(test_dir,'up', '1_oppose', name)).convert('L')
        gt = Image.open(os.path.join(test_dir,'up', '2_object/test', name)).convert('L')
        rio = Image.open(os.path.join(test_dir,'up', '10_rio', name)).convert('L')
        oppose = transforms.ToTensor()(oppose)
        gt = transforms.ToTensor()(gt)
        rio = transforms.ToTensor()(rio)
        mask_image, mask = add_mask(gt, rio)

        inp = mask_image
        # mask = Image.open(os.path.join(test_dir,'up', '13_new_mask', name)).convert('L')
        # mask = transforms.ToTensor()(mask)
        # mask[mask != 0] = 1
        mask = torch.cat((mask_add, mask), axis=0)
        oppose = oppose.unsqueeze(0)
        inp = inp.unsqueeze(0)
        mask = mask.unsqueeze(0)

        data['up']['oppose'] = oppose
        data['up']['inp'] = inp
        data['up']['mask'] = mask
        print(data['up']['mask'].shape)

        # left
        oppose = Image.open(os.path.join(test_dir,'left', '1_oppose', name)).convert('L')
        gt = Image.open(os.path.join(test_dir,'left','2_object/test', name)).convert('L')
        rio = Image.open(os.path.join(test_dir,'left', '10_rio', name)).convert('L')
        oppose = transforms.ToTensor()(oppose)
        gt = transforms.ToTensor()(gt)
        rio = transforms.ToTensor()(rio)
        mask_image, mask = add_mask(gt, rio)
        inp = mask_image
        # mask = Image.open(os.path.join(test_dir,'up', '13_new_mask', name)).convert('L')
        # mask = transforms.ToTensor()(mask)
        # mask[mask != 0] = 1
        mask = torch.cat((mask_add, mask), axis=0)
        oppose = oppose.unsqueeze(0)
        inp = inp.unsqueeze(0)
        mask = mask.unsqueeze(0)

        data['left']['oppose'] = oppose
        data['left']['inp'] = inp
        data['left']['mask'] = mask

        # right
        oppose = Image.open(os.path.join(test_dir,'right', '1_oppose', name)).convert('L')
        gt = Image.open(os.path.join(test_dir,'right', '2_object/test', name)).convert('L')
        rio = Image.open(os.path.join(test_dir,'right', '10_rio', name)).convert('L')
        oppose = transforms.ToTensor()(oppose)
        gt = transforms.ToTensor()(gt)
        rio = transforms.ToTensor()(rio)
        mask_image, mask = add_mask(gt, rio)
        inp = mask_image
        # mask = Image.open(os.path.join(test_dir,'up', '13_new_mask', name)).convert('L')
        # mask = transforms.ToTensor()(mask)
        # mask[mask != 0] = 1
        mask = torch.cat((mask_add, mask), axis=0)
        oppose = oppose.unsqueeze(0)
        inp = inp.unsqueeze(0)
        mask = mask.unsqueeze(0)

        data['right']['oppose'] = oppose
        data['right']['inp'] = inp
        data['right']['mask'] = mask


        up,left,right = validate_multi(data,generator)
        mask_up = data['up']['mask'].squeeze().numpy()
        mask_up = mask_up[1]*255
        mask_up_image = Image.fromarray(mask_up.astype(np.uint8),'L')

        mask_left = data['left']['mask'].squeeze().numpy()
        mask_left = mask_left[1] * 255
        mask_left_image = Image.fromarray(mask_left.astype(np.uint8), 'L')

        mask_right = data['right']['mask'].squeeze().numpy()
        mask_right = mask_right[1] * 255
        mask_right_image = Image.fromarray(mask_right.astype(np.uint8), 'L')


        gene_up = up * 255
        gene_left = left * 255
        gene_right = right * 255

        gene_up[gene_up < 0] = 0
        gene_left[gene_left < 0] = 0
        gene_right[gene_right < 0] = 0

        gene_up[gene_up > 255] = 255
        gene_left[gene_left > 255] = 255
        gene_right[gene_right > 255] = 255

        up_image = Image.fromarray(gene_up.astype(np.uint8), 'L')
        left_image = Image.fromarray(gene_left.astype(np.uint8), 'L')
        right_image = Image.fromarray(gene_right.astype(np.uint8), 'L')

        mask_up_image.save(save_up_mask_path)
        mask_left_image.save(save_left_mask_path)
        mask_right_image.save(save_right_mask_path)
        up_image.save(save_up_path)
        left_image.save(save_left_path)
        right_image.save(save_right_path)


