from __future__ import print_function, absolute_import, division
import json
import numpy as np
import torch
import tqdm
import cv2
from torchvision.utils import save_image
import argparse
import os
from dataset_kg import CPDataset, CPDataLoader
import torchvision.transforms as transforms

from models.semgcn import GCN_2
from common.graph_utils import adj_mx_from_edges
import torch.backends.cudnn as cudnn
import math

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

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')
    datamode = 'test'
    # Data Arguments
    parser.add_argument('--dataroot', type=str, default='../data/zalando-hd-resize', help='dataset root')
    parser.add_argument('--checkpoint_dir', type=str, default='../ckpt/kg/step_299999.pt', help='pretrained keypoints generator checkpoints')  
    parser.add_argument('--ck_point_dir', type=str, default='../example/generate_kg/{}/demo-ck-point/'.format(datamode), help='save dir of predicted cloth keypoints')
    parser.add_argument('--ck_vis_dir', type=str, default='../example/generate_kg/{}/demo-ck-vis/'.format(datamode), help='save dir of visualizations of predicted cloth keypoints')
    parser.add_argument('--sk_vis_dir', type=str, default='../example/generate_kg/{}/demo-sk-vis/'.format(datamode), help='save dir of visualizations of skeleton keypoints')
    parser.add_argument('--up', type=bool, default=True, help='whether the person and cloth images are unpaired')
    parser.add_argument('--test_list', type=str, default='demo_unpaired_pairs.txt')
    # Model Arguments
    parser.add_argument('-et', '--edge_type', type=str, default='cs', help='edge type')
    parser.add_argument('-z', '--hid_dim', type=int, default=160, help='num of hidden dimensions')
    parser.add_argument('--fine_height', type=int, default=1024, help='height of images')
    parser.add_argument('--fine_width', type=int, default=768, help='width of images')
    # Test Arguments
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--workers', type=int, default=2, help='num of workers for data loading')
    args = parser.parse_args()
    return args


def test(args):
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
    
    network = GCN_2(adj_c, adj_s, 160).to(device)
    network.load_state_dict(torch.load(args.checkpoint_dir))
    network.eval()

    val_dataset = CPDataset(args, datamode='test', data_list=args.test_list)
    val_loader = CPDataLoader(args, val_dataset, False)
    
    if not os.path.exists(args.ck_point_dir):
        os.makedirs(args.ck_point_dir)

    if not os.path.exists(args.ck_vis_dir):
        os.makedirs(args.ck_vis_dir)

    if not os.path.exists(args.sk_vis_dir):
        os.makedirs(args.sk_vis_dir)

    with torch.no_grad():
        for cnt in tqdm.tqdm(range(val_dataset.__len__()//args.batch_size)):
            inputs = val_loader.next_batch()
            s_pos = inputs['s_pos'].cuda().float()
            c_pos = inputs['c_pos'].cuda().float()
            if args.up:
                save_name = inputs['mix_name']
            else:
                save_name = inputs['im_name']
            p_pos = network(c_pos, s_pos)
            for i in range(args.batch_size):
                p_pos = p_pos.cpu()
                data = {}
                data['keypoints'] = p_pos[i].tolist()
     
                keypoints_path = os.path.join(args.ck_point_dir, save_name[i].replace('.jpg', '.json'))
                with open(keypoints_path, "w") as outfile:
                    json.dump(data, outfile)

                ck_vis_path = os.path.join(args.ck_vis_dir, save_name[i])
                estimated_cloth = draw_cloth(p_pos[i])
                save_image(estimated_cloth, ck_vis_path)

                sk_vis_path = os.path.join(args.sk_vis_dir, save_name[i])
                skeleton = draw_skeleton(s_pos[i].cpu())
                save_image(skeleton, sk_vis_path)

def main():
    args = parse_args()
    print(args)
    test(args)
    

if __name__ == "__main__":
    main()