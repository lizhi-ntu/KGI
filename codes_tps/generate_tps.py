import os
import cv2
import numpy as np
import torch
from dataset_tps import CPDataset
from torchvision.transforms import ToTensor, ToPILImage
import argparse
import tqdm
import pyclipper

class TPS(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, Y, w, h):

        """ grid """
        grid = torch.ones(1, h, w, 2)
        grid[:, :, :, 0] = torch.linspace(-1, 1, w)
        grid[:, :, :, 1] = torch.linspace(-1, 1, h)[..., None]
        grid = grid.view(-1, h * w, 2).cuda()

        """ W, A """
        n, k = X.shape[:2]
        X = X.cuda()
        Y = Y.cuda()
        Z = torch.zeros(1, k + 3, 2).cuda()
        P = torch.ones(n, k, 3).cuda()
        L = torch.zeros(n, k + 3, k + 3).cuda()

        eps = 1e-9
        D2 = torch.pow(X[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
        K = D2 * torch.log(D2 + eps)

        P[:, :, 1:] = X
        Z[:, :k, :] = Y
        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)

        Q = torch.solve(Z, L)[0]
        W, A = Q[:, :k], Q[:, k:]

        """ U """
        eps = 1e-9
        D2 = torch.pow(grid[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
        U = D2 * torch.log(D2 + eps)

        """ P """
        n, k = grid.shape[:2]
        P = torch.ones(n, k, 3).cuda()
        P[:, :, 1:] = grid

        # grid = P @ A + U @ W
        grid = torch.matmul(P, A) + torch.matmul(U, W)
        return grid.view(-1, h, w, 2)

def dedup(source_pts, target_pts, source_center, target_center):
    old_source_pts = source_pts.tolist()
    old_target_pts = target_pts.tolist()
    idx_list = []
    new_source_pts = []
    new_target_pts = []
    for idx in range(len(old_source_pts)):
        if old_source_pts[idx] not in new_source_pts:
            if old_target_pts[idx] not in new_target_pts:
                new_source_pts.append(old_source_pts[idx])
                new_target_pts.append(old_target_pts[idx])
                idx_list.append(idx)
    
    if len(idx_list) == 2:
        new_source_pts = torch.cat([source_pts[idx_list], source_center], dim=0)[None, ...]
        new_target_pts = torch.cat([target_pts[idx_list], target_center], dim=0)[None, ...]

    elif len(idx_list) > 2:
        new_source_pts = source_pts[idx_list][None, ...]
        new_target_pts = target_pts[idx_list][None, ...]
    
    else:
        print("Less than 2 points are detected !")
    
    return new_source_pts, new_target_pts

def equidistant_zoom_contour(contour, margin):
    pco = pyclipper.PyclipperOffset()
    contour = contour[:, :]
    pco.AddPath(contour, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    solution = pco.Execute(margin)
    if len(solution) == 0:
        solution = np.zeros((3, 2)).astype(int)
    else:
        solution = np.array(solution[0]).reshape(-1, 2).astype(int)

    return solution

def remove_background(args, s_mask, im):
    r_mask = s_mask.copy()
    for i in range(args.fine_height):
        for j in range(args.fine_width):
            if im[i, j, 0] >240 and im[i, j, 1] >240 and im[i, j, 2] >240:
                r_mask[i, j] = 0
    return r_mask

def draw_part(args, group_id, ten_source, ten_target, ten_source_center, ten_target_center, ten_img):
    ten_img = ten_img.cuda()
    ten_source_p = ten_source[group_id]
    ten_target_p = ten_target[group_id]
    poly = ten_target[group_id].numpy()
    poly[:, 0] = (poly[:, 0] * 0.5 + 0.5) * args.fine_width
    poly[:, 1] = (poly[:, 1] * 0.5 + 0.5) * args.fine_height
    # print(poly)
    new_poly = equidistant_zoom_contour(poly, args.margin)
    # print(new_poly)
    l_mask = np.zeros((args.fine_height, args.fine_width))
    s_mask = np.zeros((args.fine_height, args.fine_width))
    
    cv2.fillPoly(l_mask, np.int32([poly]), 255)
    cv2.fillPoly(s_mask, np.int32([new_poly]), 255)

    tps = TPS()
    ten_source_p, ten_target_p = dedup(ten_source_p, ten_target_p, ten_source_center, ten_target_center)
    warped_grid = tps(ten_target_p, ten_source_p, args.fine_width, args.fine_height)
    ten_wrp = torch.grid_sampler_2d(ten_img[None, ...], warped_grid, 0, 0, False)
    out_img = np.array(ToPILImage()(ten_wrp[0].cpu()))
    r_mask = remove_background(args, s_mask, out_img)

    return out_img, l_mask, s_mask, r_mask

def paste_cloth(mask, image, tps_image, l_mask, r_mask, parse_13):
    out_image = image.copy()
    out_mask = mask.copy()
    l_mask[(parse_13[3]).numpy() == 0] = 0
    r_mask[(parse_13[3]).numpy() == 0] = 0

    out_mask[l_mask==255] = 0
    out_mask[r_mask==255] = 255
    
    out_image[l_mask==255, :] = 0
    out_image[r_mask==255, :] = tps_image[r_mask==255, :]

    return out_mask, out_image

def generate_repaint(args, image, cloth, source, target, ag_mask, skin_mask, parse_13):
    out_mask = ag_mask.copy()
    out_image = image.copy()
    out_image[ag_mask==0, :] = 0
    
    # paste skin 
    new_skin_mask = skin_mask.copy()
    new_skin_mask[(parse_13[5] + parse_13[6] + parse_13[11]).numpy() == 0] = 0

    out_mask[new_skin_mask==255] = 255
    out_image[new_skin_mask==255, :] = image[new_skin_mask==255, :]
    
    # paste cloth
    group_backbone =  [ 4,  3,  2,  1,  0,  5, 14, 15, 16, 17, 18, 19, 20, 21, 22, 31]
    group_left_up  =  [ 5,  6,  7, 12, 13, 14]
    group_left_low =  [ 7,  8,  9, 10, 11, 12]
    group_right_up =  [22, 23, 24, 29, 30, 31]
    group_right_low = [24, 25, 26, 27, 28, 29]

    ten_cloth = ToTensor()(cloth)

    ten_source = (source - 0.5) * 2
    ten_target = (target - 0.5) * 2
    ten_source_center = (0.5 * (ten_source[18] - ten_source[2]))[None, ...] # [B x NumPoints x 2]
    ten_target_center = (0.5 * (ten_target[18] - ten_target[2]))[None, ...] # [B x NumPoints x 2]

    # Whole Points TPS
    im_backbone, l_mask_backbone, s_mask_backbone, r_mask_backbone = draw_part(
        args, group_backbone, ten_source, ten_target, ten_source_center, ten_target_center, ten_cloth)
    im_left_up, l_mask_left_up, s_mask_left_up, r_mask_left_up = draw_part(
        args, group_left_up, ten_source, ten_target, ten_source_center, ten_target_center, ten_cloth)
    im_right_up, l_mask_right_up, s_mask_right_up, r_mask_right_up = draw_part(
        args, group_right_up, ten_source, ten_target, ten_source_center, ten_target_center, ten_cloth)
    im_left_low, l_mask_left_low, s_mask_left_low, r_mask_left_low = draw_part(
        args, group_left_low, ten_source, ten_target, ten_source_center, ten_target_center, ten_cloth)
    im_right_low, l_mask_right_low, s_mask_right_low, r_mask_right_low = draw_part(
        args, group_right_low, ten_source, ten_target, ten_source_center, ten_target_center, ten_cloth)
    
    if r_mask_backbone.sum() / s_mask_backbone.sum() < 0.9:
        r_mask_backbone = s_mask_backbone
    
    out_mask, out_image = paste_cloth(out_mask, out_image, im_backbone, l_mask_backbone, r_mask_backbone, parse_13)
    out_mask, out_image = paste_cloth(out_mask, out_image, im_left_up, l_mask_left_up, r_mask_left_up, parse_13)
    out_mask, out_image = paste_cloth(out_mask, out_image, im_left_low, l_mask_left_low, r_mask_left_low, parse_13)
    out_mask, out_image = paste_cloth(out_mask, out_image, im_right_up, l_mask_right_up, r_mask_right_up, parse_13)
    out_mask, out_image = paste_cloth(out_mask, out_image, im_right_low, l_mask_right_low, r_mask_right_low, parse_13)
    
    return out_image, out_mask

def generation_tps(args):
    if not os.path.exists(args.save_dir_image):
        os.makedirs(args.save_dir_image)
    if not os.path.exists(args.save_dir_mask):
        os.makedirs(args.save_dir_mask)

    test_dataset = CPDataset(args, datamode=args.datamode, data_list=args.data_list, up=args.up)

    for p in tqdm.tqdm(range(0, test_dataset.__len__())):
        inputs = test_dataset.__getitem__(p)
        image = inputs['image']
        cloth = inputs['cloth']
        ag_mask = inputs['ag_mask']
        skin_mask = inputs['skin_mask']
        parse_13 = inputs['parse_13']
        source = inputs['v_pos'].float()
        target = inputs['e_pos'].float()
        
        out_image, out_mask = generate_repaint(args, image, cloth, source, target, ag_mask, skin_mask, parse_13)
        save_name = inputs['mix_name'].replace('.jpg', '.png')
        cv2.imwrite(os.path.join(args.save_dir_image, save_name), out_image)
        cv2.imwrite(os.path.join(args.save_dir_mask, save_name), out_mask)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')
    # Data Arguments
    parser.add_argument('--datamode', type=str, default='test')
    parser.add_argument('--dataroot', type=str, default='../data/zalando-hd-resize')
    parser.add_argument('--data_list', type=str, default='test_paired_pairs.txt')
    parser.add_argument('--up', type=bool, default=False)
    parser.add_argument('--parse_name', type=str, default='paired-full-parse-mix')
    parser.add_argument('--fine_height', type=int, default=256*4)
    parser.add_argument('--fine_width', type=int, default=192*4)
    parser.add_argument('--margin', type=int, default=-5)
    parser.add_argument('--save_dir_mask', type=str, default='../example/generate_repaint/test/mask_paired_mix_1024')
    parser.add_argument('--save_dir_image', type=str, default='../example/generate_repaint/test/image_paired_mix_1024')
    args = parser.parse_args()
    
    if args.fine_height == 256:
        args.margin = -2
    elif args.fine_height == 512:
        args.margin = -3
    elif args.fine_height == 1024:
        args.margin = -5

    return args

if __name__=='__main__':
    args = parse_args()
    print(args)
    generation_tps(args)
