import torch
import numpy as np
from PIL import Image
import argparse
import os
from dataset_pg import CPDataset, CPDataLoader
from networks_pg import ParseGenerator
from tqdm import tqdm
from utils_pg import *

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

def get_opt():
    datamode = 'test'
    parser = argparse.ArgumentParser()
    # Data Arguments
    parser.add_argument('--exp_name', type=str, default='parse_full')
    parser.add_argument('--parse_ag_mode', type=str, default='parse_ag_full')
    parser.add_argument('--datamode', type=str, default=datamode)
    parser.add_argument('--dataroot', type=str, default='../data/zalando-hd-resize')
    parser.add_argument('--data_list', type=str, default='test_paired_pairs.txt')
    parser.add_argument('--lenth', type=int, default=2032)
    parser.add_argument('--up', type=bool, default=False)
    parser.add_argument('--fine_width', type=int, default=192*4)
    parser.add_argument('--fine_height', type=int, default=256*4)
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/mix/parse_ag_full/step_9999.pt')
    parser.add_argument('--vis_save_dir', type=str, default='../example/generate_pg_mix_full/{}/paired-full-parse/'.format(datamode))
    # Model Arguments
    parser.add_argument('--semantic_nc', type=int, default=13)
    parser.add_argument('--output_nc', type=int, default=13)
    # Train Arguments
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)

    opt = parser.parse_args()
    return opt

def test(opt, network, eval_loader):
    network.eval()
    
    if not os.path.exists(opt.vis_save_dir):
        os.makedirs(opt.vis_save_dir)

    with torch.no_grad():
        for cnt in tqdm(range(opt.lenth//opt.batch_size)):
            inputs = eval_loader.next_batch()
            mix_name = inputs['mix_name']
            sk_vis = inputs['sk_vis'].cuda()
            ck_vis = inputs['ck_vis'].cuda()
            parse_ag = inputs['parse_ag'].cuda()

            ins = torch.cat([parse_ag, sk_vis, ck_vis], 1)
            preds = network(ins)                

            preds = pred_to_onehot(preds)
            for i in range(opt.batch_size):
                out_im = visualize_segmap(preds.cpu(), tensor_out=False, batch=i)
                out_im = out_im.resize((opt.fine_width, opt.fine_height))
                out_im.save(os.path.join(opt.vis_save_dir, mix_name[i]).replace('.jpg', '.png'))
                
def main():
    opt = get_opt()
    print(opt)

    network = ParseGenerator(input_nc=19, output_nc=opt.output_nc, ngf=64).to(torch.device('cuda'))
    network.load_state_dict(torch.load(opt.checkpoint))
    test_dataset = CPDataset(opt, datamode=opt.datamode, data_list=opt.data_list, up=opt.up)
    opt.lenth = test_dataset.lenth
    test_loader = CPDataLoader(opt, test_dataset, False)
    test(opt, network, test_loader)

if __name__ == "__main__":
    main()