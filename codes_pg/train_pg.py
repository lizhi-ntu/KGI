import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid
from torchvision.utils import save_image
import argparse
import os
from dataset_pg import CPDataset, CPDataLoader
from networks_pg import ParseGenerator
from tqdm import tqdm
from PIL import Image
from utils_pg import *
from loss_pg import SSIM, IOU
import torchvision.transforms as transforms


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

def visualize_segmap(input, multi_channel=True, tensor_out=True, batch=0) :
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

def train(opt):
    network = ParseGenerator(input_nc=19, output_nc=opt.output_nc, ngf=64)
    
    train_dataset = CPDataset(opt, datamode='train', data_list=opt.train_list, up=False)
    paired_test_dataset = CPDataset(opt, datamode='test', data_list=opt.paired_test_list, up=False)
    unpaired_test_dataset = CPDataset(opt, datamode='test', data_list=opt.unpaired_test_list, up=True)

    paired_test_lenth = paired_test_dataset.__len__()
    unpaired_test_lenth = unpaired_test_dataset.__len__()
    
    train_loader = CPDataLoader(opt, train_dataset, True)
    paired_test_loader = CPDataLoader(opt, paired_test_dataset, False)
    unpaired_test_loader = CPDataLoader(opt, unpaired_test_dataset, False)

    network.train()
    network.cuda()
    
    criterion_BCE = nn.BCELoss(size_average=True)
    criterion_SSIM = SSIM(window_size=11,size_average=True)
    criterion_IOU = IOU(size_average=True)
    optimizer_G = torch.optim.Adam(network.parameters(), lr=opt.G_lr, betas=(0.5, 0.999))
    
    idx = tqdm(range(opt.load_step, opt.keep_step))
    total_loss = 0
    for step in idx:
        inputs = train_loader.next_batch()
        
        sk_vis = inputs['sk_vis'].cuda()
        ck_vis = inputs['ck_vis'].cuda()
        parse_agnostic = inputs['parse_ag'].cuda()
        parse = inputs['parse'].cuda()
        
        ins = torch.cat([parse_agnostic, sk_vis, ck_vis], 1)
        preds = network(ins)
        
        if opt.loss == 'bce':
            loss = criterion_BCE(preds, parse)
        else:
            bce_out = criterion_BCE(preds, parse)
            ssim_out = 1 - criterion_SSIM(preds, parse)
            iou_out = criterion_IOU(preds, parse)
            loss = bce_out + ssim_out + iou_out

        total_loss = total_loss + loss / opt.test_count
        
        idx.set_description((
            f"model_id:{step};"
            f"total loss:{total_loss.item():.5f};"
            f"batch loss:{loss.item():.5f};"
            ))
        
        optimizer_G.zero_grad()
        loss.backward()
        optimizer_G.step()
        
        if (step + 1) % opt.test_count == 0:
            if not os.path.exists(os.path.join(opt.vis_save_dir, opt.exp_name, 'paired', 'step_{}'.format(step))):
                os.makedirs(os.path.join(opt.vis_save_dir, opt.exp_name, 'paired', 'step_{}'.format(step)))
            
            if not os.path.exists(os.path.join(opt.vis_save_dir, opt.exp_name, 'unpaired', 'step_{}'.format(step))):
                os.makedirs(os.path.join(opt.vis_save_dir, opt.exp_name, 'unpaired', 'step_{}'.format(step)))

            if not os.path.exists(os.path.join(opt.checkpoint_dir, opt.exp_name)):
                os.makedirs(os.path.join(opt.checkpoint_dir, opt.exp_name))

            network.eval()
            with torch.no_grad():
                for cnt in tqdm(range(paired_test_lenth//4)):
                    inputs = paired_test_loader.next_batch()
                    mix_name = inputs['mix_name']
                    sk_vis = inputs['sk_vis'].cuda()
                    ck_vis = inputs['ck_vis'].cuda()
                    parse_agnostic = inputs['parse_ag'].cuda()
                    parse = inputs['parse'].cuda()

                    ins = torch.cat([parse_agnostic, sk_vis, ck_vis], 1)
                    preds = network(ins)
                         
                    preds = pred_to_onehot(preds)
                    for i in range(opt.batch_size):
                        grid = make_grid([
                            visualize_segmap(parse_agnostic.cpu(), batch=i),
                            (sk_vis[i].cpu()/2 +0.5), 
                            (ck_vis[i].cpu()/2 + 0.5),
                            visualize_segmap(preds.cpu(), batch=i),
                            visualize_segmap(parse.cpu(), batch=i),
                            ], nrow=5)

                        grid_name = os.path.join(opt.vis_save_dir, opt.exp_name, 'paired', 'step_{}'.format(step), mix_name[i])
                        save_image(grid, grid_name)
                
                for cnt in tqdm(range(unpaired_test_lenth//4)):
                    inputs = unpaired_test_loader.next_batch()
                    mix_name = inputs['mix_name']
                    sk_vis = inputs['sk_vis'].cuda()
                    ck_vis = inputs['ck_vis'].cuda()
                    parse_agnostic = inputs['parse_ag'].cuda()

                    ins = torch.cat([parse_agnostic, sk_vis, ck_vis], 1)
                    preds = network(ins)
                         
                    preds = pred_to_onehot(preds)
                    for i in range(opt.batch_size):
                        grid = make_grid([
                            visualize_segmap(parse_agnostic.cpu(), batch=i),
                            (sk_vis[i].cpu()/2 +0.5), 
                            (ck_vis[i].cpu()/2 + 0.5),
                            visualize_segmap(preds.cpu(), batch=i),
                            ], nrow=4)
                        grid_name = os.path.join(opt.vis_save_dir, opt.exp_name, 'unpaired', 'step_{}'.format(step), mix_name[i])
                        save_image(grid, grid_name)

            torch.save(network.state_dict(), os.path.join(opt.checkpoint_dir, opt.exp_name, 'step_{}.pt'.format(step)))
            print(total_loss.item())
            total_loss = 0
            network.train()

def get_opt():
    parser = argparse.ArgumentParser()
    # Data Arguments
    parser.add_argument('--exp_name', type=str, default='parse_full')
    parser.add_argument('--parse_ag_mode', type=str, default='parse_ag_full')
    parser.add_argument('--dataroot', default='../data/zalando-hd-resized')
    parser.add_argument('--fine_width', type=int, default=192*4)
    parser.add_argument('--fine_height', type=int, default=256*4)
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints/')
    parser.add_argument('--vis_save_dir', type=str, default='../visualizations/')
    parser.add_argument('--train_list', default='train_pairs.txt')
    parser.add_argument('--paired_test_list', default='demo_paired_pairs.txt')
    parser.add_argument('--unpaired_test_list', default='demo_unpaired_pairs.txt')
    parser.add_argument('--load_step', type=int, default=0)
    parser.add_argument('--keep_step', type=int, default=20000)
    # Model Arguments
    parser.add_argument('--semantic_nc', type=int, default=13)
    parser.add_argument('--output_nc', type=int, default=13)
    # Train Arguments
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--loss', type=str, default='mix')
    parser.add_argument('--G_lr', type=float, default=0.0002, help='Generator initial learning rate for adam')
    parser.add_argument('--test_count', type=int, default=5000)
    
    opt = parser.parse_args()
    return opt

def main():
    opt = get_opt()
    print(opt)
    train(opt)

if __name__ == "__main__":
    main()