from __future__ import print_function, absolute_import, division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import cv2
from torchvision.utils import make_grid
from torchvision.utils import save_image
import argparse
import os
from dataset_kg import CPDataset, CPDataLoader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from models.semgcn import GCN_2
from common.graph_utils import adj_mx_from_edges
import math

def edge_loss(t_pos, p_pos, edges_c):
    edges = torch.tensor(edges_c).cuda()
    line_t = t_pos[:, edges[:, 1], :] - t_pos[:, edges[:, 0], :]
    line_p = p_pos[:, edges[:, 1], :] - p_pos[:, edges[:, 0], :]
    
    eloss = 1 - F.cosine_similarity(line_t, line_p, dim=2).mean(1).mean(0)
    return eloss

def draw_skeleton(sk_pos):
    sk_pos[:, 0] = sk_pos[:, 0] * 768
    sk_pos[:, 1] = sk_pos[:, 1] * 1024

    sk_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sk_Seq = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], [1,9], [8,9]]

    stickwidth = 10

    jk_colors = [[255, 85, 0], [0, 255, 255], [255, 170, 0], [255, 255, 0], [255, 255, 0], [255, 170, 0], [85, 255, 0], [85, 255, 0], [0, 170, 255], [0, 170, 255]]
    sk_colors = [[255, 85, 0], [255, 170, 0], [255, 255, 0], [255, 255, 0], [255, 170, 0], [85, 255, 0], \
	          [85, 255, 0], [0, 255, 255], [0, 255, 255], [0, 170, 255]]

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
    # Data Arguments
    parser.add_argument('--exp_name', type=str, default='two_graph_cs', help='name of the experiment')
    parser.add_argument('--dataroot', type=str, default='../data/zalando-hd-resize', help='dataset root')
    parser.add_argument('--train_list', type=str, default='train_pairs.txt', help='training data list')
    parser.add_argument('--paired_test_list', type=str, default='demo_paired_pairs.txt', help='paired validation data list')
    parser.add_argument('--unpaired_test_list', type=str, default='demo_unpaired_pairs.txt', help='unpaired validation data list')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints', help='save dir of checkpoints')
    parser.add_argument('--log_dir', type=str, default='../logs', help='save dir of logs')
    parser.add_argument('--vis_save_dir', type=str, default='../visualizations/', help='save dir of visualizations')
    # Model Arguments
    parser.add_argument('-et', '--edge_type', type=str, default='cs', help='edge type of the network')
    parser.add_argument('-z', '--hid_dim', type=int, default=160, help='num of hidden dimensions')
    parser.add_argument('--fine_height', type=int, default=1024, help='height of image frame')
    parser.add_argument('--fine_width', type=int, default=768, help='width of image frame')
    # Train Arguments
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR', help='initial learning rate')
    parser.add_argument('--load_step', type=int, default=0, help='intial of training steps')
    parser.add_argument('--keep_step', type=int, default=300000, help='maximum of training steps')
    parser.add_argument('--test_count', type=int, default=100, help='number of steps for testing')
    parser.add_argument('--lamda', default=1, type=float, help='weight for the edge loss')
    parser.add_argument('--workers', default=4, type=int, help='num of workers')
    #=======================================================
    
    args = parser.parse_args()
    return args

def train(args):
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
    criterion1 = nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    
    train_dataset = CPDataset(args, datamode='train', data_list=args.train_list)
    paired_test_dataset = CPDataset(args, datamode='test', data_list=args.paired_test_list)
    unpaired_test_dataset = CPDataset(args, datamode='test', data_list=args.unpaired_test_list, up=True)
   
    train_loader = CPDataLoader(args, train_dataset, True)
    paired_test_loader = CPDataLoader(args, paired_test_dataset, False)
    unpaired_test_loader = CPDataLoader(args, unpaired_test_dataset, False)
    
    network.train()
    total_loss = 0
    idx = tqdm.tqdm(range(args.load_step, args.keep_step))
    for step in idx:
        inputs = train_loader.next_batch()
        s_pos = inputs['s_pos'].cuda().float()
        c_pos = inputs['c_pos'].cuda().float()
        t_pos = inputs['t_pos'].cuda().float()
        
        p_pos = network(c_pos, s_pos)           
        
        #=====================================================================================================
        loss = criterion1(p_pos, t_pos) + args.lamda * edge_loss(p_pos, t_pos, edges_c)
        #=====================================================================================================

        total_loss = total_loss + loss.item()
        idx.set_description((
            f"model_id:{step};"
            f"batch loss:{loss.item():.5f};"))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % args.test_count == 0:
            if not os.path.exists(os.path.join(args.vis_save_dir, args.exp_name, 'paired', 'step_{}'.format(step))):
                os.makedirs(os.path.join(args.vis_save_dir, args.exp_name, 'paired', 'step_{}'.format(step)))

            if not os.path.exists(os.path.join(args.vis_save_dir, args.exp_name, 'unpaired', 'step_{}'.format(step))):
                os.makedirs(os.path.join(args.vis_save_dir, args.exp_name, 'unpaired', 'step_{}'.format(step)))

            if not os.path.exists(os.path.join(args.checkpoint_dir, args.exp_name)):
                os.makedirs(os.path.join(args.checkpoint_dir, args.exp_name))

            if not os.path.exists(os.path.join(args.log_dir, args.exp_name)):
                os.makedirs(os.path.join(args.log_dir, args.exp_name))

            with open('{}/{}.txt'.format(args.log_dir, args.exp_name), 'a') as f:
                f.write('\nStep: {}  '.format(step))
                f.write('total loss: {total_loss:.5f}'.format(total_loss=total_loss))
            
            total_loss = 0
            network.eval()
            with torch.no_grad():
                for cnt in tqdm.tqdm(range(96//args.batch_size)):
                    inputs = paired_test_loader.next_batch()
                    im_name = inputs['im_name']
                    s_pos = inputs['s_pos'].cuda().float()
                    c_pos = inputs['c_pos'].cuda().float()
                    t_pos = inputs['t_pos'].cuda().float()
                    v_pos = inputs['v_pos'].cuda().float()
                    
                    p_pos = network(c_pos, s_pos)                     

                    for i in range(args.batch_size):
                        grid = make_grid([
                            draw_skeleton(s_pos[i].cpu()),
                            draw_cloth(v_pos[i].cpu()),
                            draw_cloth(p_pos[i].cpu()),
                            draw_cloth(t_pos[i].cpu())
                            ], nrow=4)
                        grid_name = os.path.join(args.vis_save_dir, args.exp_name, 'paired', 'step_{}'.format(step), im_name[i])
                        save_image(grid, grid_name)

                for cnt in tqdm.tqdm(range(16//args.batch_size)):
                    inputs = unpaired_test_loader.next_batch()
                    im_name = inputs['mix_name']
                    s_pos = inputs['s_pos'].cuda().float()
                    c_pos = inputs['c_pos'].cuda().float()
                    v_pos = inputs['v_pos'].cuda().float()
                    
                    p_pos = network(c_pos, s_pos)                     

                    for i in range(args.batch_size):
                        grid = make_grid([
                            draw_skeleton(s_pos[i].cpu()),
                            draw_cloth(v_pos[i].cpu()),
                            draw_cloth(p_pos[i].cpu()),
                            ], nrow=3)
                        grid_name = os.path.join(args.vis_save_dir, args.exp_name, 'unpaired', 'step_{}'.format(step), im_name[i])
                        save_image(grid, grid_name)


            torch.save(network.state_dict(), os.path.join(args.checkpoint_dir, args.exp_name, 'step_{}.pt'.format(step)))
        network.train()

def main():
    args = parse_args()
    print(args)
    train(args)
    

if __name__ == "__main__":
    main()