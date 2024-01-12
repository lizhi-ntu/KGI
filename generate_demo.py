from __future__ import print_function, absolute_import, division
import math
import numpy as np
import torch
import tqdm
import cv2
import argparse
import os
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.utils import save_image, make_grid
from dataset_demo import CPDataset, CPDataLoader
import torchvision.transforms as transforms
import pyclipper

from codes_kg.models.semgcn import GCN_2
from codes_kg.common.graph_utils import adj_mx_from_edges

from codes_pg.networks_pg import ParseGenerator
from codes_pg.utils_pg import *

from codes_demo.conf_mgt import conf_base
from codes_demo.utils import yamlread
from codes_demo.guided_diffusion import dist_util
from PIL import Image
from codes_demo.guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion, select_args

def draw_skeleton(sk_pos):
    sk_pos[:, 0] = sk_pos[:, 0] * 768
    sk_pos[:, 1] = sk_pos[:, 1] * 1024

    sk_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sk_Seq = [[0,1], [1,8], [1,9], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7]]

    stickwidth = 10

    jk_colors = [[255, 85, 0], [0, 255, 255], [255, 170, 0], [255, 255, 0], [255, 255, 0], [255, 170, 0], [85, 255, 0], [85, 255, 0], [0, 255, 255], [0, 255, 255]]
    sk_colors = [[255, 85, 0], [0, 255, 255], [0, 255, 255], [255, 170, 0], [255, 255, 0], [255, 255, 0], [255, 170, 0], [85, 255, 0], \
	          [85, 255, 0]]

    canvas = np.zeros((1024,768,3),dtype = np.uint8) # B,G,R order

    for i in range(len(sk_idx)):
        cv2.circle(canvas, (int(sk_pos[sk_idx[i]][0]),int(sk_pos[sk_idx[i]][1])), stickwidth, jk_colors[i], thickness=-1)
    
    for i in range(len(sk_Seq)):
        index = np.array(sk_Seq[i])
        cur_canvas = canvas.copy()
        Y = [sk_pos[index[0]][0],sk_pos[index[1]][0]]
        X = [sk_pos[index[0]][1],sk_pos[index[1]][1]]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, sk_colors[i])
        canvas = cv2.addWeighted(canvas, 0, cur_canvas, 1, 0)
    
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    transform = transforms.ToTensor()
    canvas = transform(canvas)
    
    return canvas

def draw_cloth(ck_pos):
    ck_pos[:, 0] = ck_pos[:, 0] * 768
    ck_pos[:, 1] = ck_pos[:, 1] * 1024

    canvas = np.zeros((1024,768,3),dtype = np.uint8) # B,G,R order

    ck_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    ck_Seq = [[0,1], [1,2], [2,3], [3,4], \
        [4,31], [31,30], [30,29], [29,28], [28,27], [27,26], [26,25], [25,24], [24,23], [23,22], [22,21], [21,20], [20,19],\
        [19,18], [18,17], [17,16], [16,15], [15,14], [14,13], [13,12], [12,11], [11,10], [10,9], [9,8], [8,7], [7,6], [6,5], [5,0]]

    stickwidth = 10
    ck_colors = [255, 0, 0]

    for i in ck_idx:
        cv2.circle(canvas, (int(ck_pos[i][0]),int(ck_pos[i][1])), stickwidth, ck_colors, thickness=-1)
    
    for i in range(len(ck_Seq)):
        index = np.array(ck_Seq[i])
        cur_canvas = canvas.copy()
        Y = [ck_pos[index[0]][0], ck_pos[index[1]][0]]
        X = [ck_pos[index[0]][1], ck_pos[index[1]][1]]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, ck_colors)
        canvas = cv2.addWeighted(canvas, 0, cur_canvas, 1, 0)
    
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    transform = transforms.ToTensor()
    canvas = transform(canvas)
    
    return canvas

def pred_to_onehot(prediction):
    size = prediction.shape
    prediction_max = torch.argmax(prediction, dim=1)
    oneHot_size = (size[0], 13, size[2], size[3])
    pred_onehot = torch.FloatTensor(torch.Size(oneHot_size)).zero_().cuda()
    pred_onehot = pred_onehot.scatter_(1, prediction_max.unsqueeze(1).data.long(), 1.0)
    return pred_onehot

def ndim_tensor2im(image_tensor, imtype=np.uint8, batch=0):
    image_numpy = image_tensor[batch].cpu().float().numpy()
    result = np.argmax(image_numpy, axis=0)
    return result.astype(imtype)

def visualize_segmap(input, multi_channel=True, tensor_out=True, batch=0):
    palette = [
        0, 0, 0, 128, 0, 0, 254, 0, 0, 0, 85, 0, 169, 0, 51,
        254, 85, 0, 0, 0, 85, 0, 119, 220, 85, 85, 0, 0, 85, 85,
        85, 51, 0, 52, 86, 128, 0, 128, 0, 0, 0, 254, 51, 169, 220,
        0, 254, 254, 85, 254, 169, 169, 254, 85, 254, 254, 0, 254, 169, 0
    ]
    input = input.detach()
    if multi_channel :
        input = ndim_tensor2im(input,batch=batch)
    else :
        input = input[batch][0].cpu()
        input = np.asarray(input)
        input = input.astype(np.uint8)
    input = Image.fromarray(input, 'P')
    input.putpalette(palette)

    if tensor_out :
        trans = transforms.ToTensor()
        return trans(input.convert('RGB'))

    return input

#====================TPS
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
    out_img = np.array(transforms.ToPILImage()(ten_wrp[0].cpu()))
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
    image_ag = out_image.copy()
    
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

    ten_cloth = transforms.ToTensor()(cloth)

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
    
    return out_image, out_mask, image_ag

def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')
    # Data Arguments
    parser.add_argument("--dataroot", type=str, default="data/zalando-hd-resize")
    parser.add_argument("--parse_ag_mode", type=str, default='parse_ag_full')
    parser.add_argument('--vis_dir', type=str, default='example/demo/vis_fp32_19999/')
    parser.add_argument('--final_results_dir', type=str, default='example/demo/final_results/')
    parser.add_argument('--kg_checkpoint_dir', type=str, default='ckpt/kg/step_299999.pt')
    parser.add_argument('--pg_checkpoint_dir', type=str, default='checkpoints/mix/parse_ag_full/step_9999.pt')
    parser.add_argument('--up', type=bool, default=True)
    parser.add_argument('--test_list', type=str, default='demo_unpaired_pairs.txt')
    # KG Model Arguments
    parser.add_argument('-et', '--edge_type', type=str, default='cs')
    parser.add_argument('-z', '--hid_dim', type=int, default=160)
    parser.add_argument('--fine_height', type=int, default=1024)
    parser.add_argument('--fine_width', type=int, default=768)
    # PG Model Arguments
    parser.add_argument("--semantic_nc", type=int, default=13)
    parser.add_argument("--output_nc", type=int, default=13)
    # TPS Arguments
    parser.add_argument('--margin', type=int, default=-5)
    # Test Arguments
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--conf_path', type=str, required=False, default='codes_sci/confs/demo.yml')
    args = parser.parse_args()
    # SCI Arguments    
    args_sci = conf_base.Default_Conf()
    args_sci.update(yamlread(args.conf_path))

    return args, args_sci


def test(args, args_sci):
    cudnn.benchmark = True
    num_pts_c = 32
    num_pts_s = 10
    device = "cuda"
    #======================================
     
    contour_edges=[
        [0, 1],
        [1, 2], 
        [2, 3], 
        [3, 4],
        [4, 31], 
        [31, 30],
        [30, 29],
        [29, 28],
        [28, 27], 
        [27, 26], 
        [26, 25], 
        [25, 24], 
        [24, 23], 
        [23, 22], 
        [22, 21], 
        [21, 20], 
        [20, 19], 
        [19, 18], 
        [18, 17], 
        [17, 16], 
        [16, 15], 
        [15, 14], 
        [14, 13], 
        [13, 12], 
        [12, 11], 
        [11, 10], 
        [10, 9], 
        [9, 8], 
        [8, 7], 
        [7, 6], 
        [6, 5], 
        [5, 0]]

    symmetry_edges=[
        [0, 4],
        [1, 3], 
        [5, 31], 
        [14, 22], 
        [15, 21], 
        [16, 20], 
        [17, 19], 
        [6, 13], 
        [7, 12], 
        [8, 11], 
        [23, 30], 
        [24, 29], 
        [25, 28],
        [2, 18]]

    edges_c = contour_edges + symmetry_edges

    edges_s=[
        [0, 1],
        [1, 2], [1, 5], 
        [2, 3], [5, 6], 
        [3, 4], [6, 7], 
        [1, 8], [1, 9]]
    
    adj_c = adj_mx_from_edges(num_pts_c, edges_c, False)
    adj_s = adj_mx_from_edges(num_pts_s, edges_s, False)
    
    kg_network = GCN_2(adj_c, adj_s, 160).to(device)
    kg_network.load_state_dict(torch.load(args.kg_checkpoint_dir))
    kg_network.eval()

    pg_network = ParseGenerator(input_nc=19, output_nc=args.output_nc, ngf=64).to(torch.device('cuda'))
    pg_network.load_state_dict(torch.load(args.pg_checkpoint_dir))
    pg_network.eval()
    
    sci_model, diffusion = create_model_and_diffusion(
        **select_args(args_sci, model_and_diffusion_defaults().keys()), conf=args_sci
    )
    sci_model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            args_sci.model_path), map_location="cpu")
    )
    sci_model.to('cuda')
    
    if args_sci.use_fp16:
        sci_model.convert_to_fp16()
    sci_model.eval()

    show_progress = args_sci.show_progress
    cond_fn = None
    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return sci_model(x, t, y)

    sample_fn = (
        diffusion.p_sample_loop if not args_sci.use_ddim else diffusion.ddim_sample_loop
    )

    totensor_transform = transforms.ToTensor()
    norm_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    val_dataset = CPDataset(args, datamode='test', data_list=args.test_list, up=args.up)
    val_loader = CPDataLoader(args, val_dataset, False)
    
    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)
    if not os.path.exists(args.final_results_dir):
        os.makedirs(args.final_results_dir)

    with torch.no_grad():
        for cnt in tqdm.tqdm(range(val_dataset.__len__())):
            
            inputs = val_loader.next_batch()
            image = inputs['image'].numpy()
            cloth = inputs['cloth'].numpy()
            ag_mask = inputs['ag_mask'].numpy()
            skin_mask = inputs['skin_mask'].numpy()
            parse = inputs['parse']
            parse_ag = inputs['parse_ag'].cuda()
            s_pos = inputs['s_pos'].cuda().float()
            c_pos = inputs['c_pos'].cuda().float()
            v_pos = inputs['v_pos'].float()

            save_name = inputs['mix_name']

            p_pos = kg_network(c_pos, s_pos)
            p_pos = p_pos.cpu()

            sk_vis = draw_skeleton(s_pos[0].detach().clone().cpu())
            ck_vis = draw_cloth(p_pos[0].detach().clone().cpu())
            vk_vis = draw_cloth(v_pos[0].detach().clone().cpu())
 
            sk_input = torch.unsqueeze(norm_transform(sk_vis), dim=0).cuda()
            ck_input = torch.unsqueeze(norm_transform(ck_vis), dim=0).cuda()
            pg_input = torch.cat([parse_ag, sk_input, ck_input], 1)
            
            pg_output = pg_network(pg_input)
            pg_output = pred_to_onehot(pg_output).cpu()
            out_image, out_mask, image_ag = generate_repaint(args, image[0], cloth[0], v_pos[0], p_pos[0], ag_mask[0], skin_mask[0], pg_output[0])

            model_kwargs = {}
            model_kwargs['gt'] = torch.unsqueeze(norm_transform(totensor_transform(cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB))).cuda(), dim=0)
            model_kwargs['gt_keep_mask'] = torch.unsqueeze(totensor_transform(cv2.cvtColor(out_mask, cv2.COLOR_GRAY2RGB)).cuda(), dim=0)
            model_kwargs['y'] = pg_output.detach().clone().cuda()

            sci_result = sample_fn(
                model_fn,
                (1, 3, args_sci.image_size, int(args_sci.image_size*0.75)),
                clip_denoised=args_sci.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=device,
                progress=show_progress,
                return_all=True,
                conf=args_sci
                )
            final_image = toU8(sci_result['sample'])

            cloth_vis = totensor_transform(cv2.cvtColor(cloth[0], cv2.COLOR_BGR2RGB))
            image_vis = totensor_transform(cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB))
            image_ag_vis = totensor_transform(cv2.cvtColor(image_ag, cv2.COLOR_BGR2RGB))
            incomplete_image_vis = totensor_transform(cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB))
            content_keeping_mask_vis = totensor_transform(cv2.cvtColor(out_mask, cv2.COLOR_GRAY2RGB))
            final_image_vis = totensor_transform(final_image[0])

            parse_vis = visualize_segmap(parse.cpu(), tensor_out=True, batch=0)
            parse_ag_vis = visualize_segmap(parse_ag.cpu(), tensor_out=True, batch=0)
            parse_estimate_vis = visualize_segmap(pg_output.cpu(), tensor_out=True, batch=0)
            
            grid = make_grid([
                cloth_vis,
                vk_vis,
                sk_vis,
                ck_vis,
                parse_vis,
                parse_ag_vis,
                content_keeping_mask_vis,
                parse_estimate_vis,
                image_vis,
                image_ag_vis,
                incomplete_image_vis,
                final_image_vis
                ], nrow=4)
            grid_path = os.path.join(args.vis_dir, save_name[0])
            save_image(grid, grid_path)


def main():
    args, args_sci = parse_args()
    print(args)
    print(args_sci)
    test(args, args_sci)
    

if __name__ == "__main__":
    main()