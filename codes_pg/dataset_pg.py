#coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os.path as osp
import numpy as np
import cv2


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
    if batch == None:
        input = input[None]
        batch = 0
    
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


class CPDataset(data.Dataset):
    """
    Dataset for Parse (Segmentation Map) Generation
    """
    def __init__(self, opt, datamode, data_list, up):
        super(CPDataset, self).__init__()
        # base setting
        self.opt = opt
        self.up = up
        self.root = opt.dataroot
        self.datamode = datamode 
        self.data_list = data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = osp.join(opt.dataroot, datamode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        # load data list
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names
        self.lenth = len(self.im_names)

    def name(self):
        return "CPDataset"

    def get_mask(self, poly):
        poly[:, 0] = (poly[:, 0]) * self.fine_width
        poly[:, 1] = (poly[:, 1]) * self.fine_height
        im_mask = np.zeros((self.fine_height, self.fine_width))
        cv2.fillPoly(im_mask, np.int32([poly]), 1)

        return im_mask

    def __getitem__(self, index):
        im_name = self.im_names[index]
        
        if self.up == True:
            c_name = self.c_names[index]
            mix_name = '{}_{}'.format(im_name.split('.')[0], c_name)
        else:
            c_name = im_name
            mix_name = im_name

        # parsing image
        parse_name = im_name.replace('.jpg', '.png')
        parse = Image.open(osp.join(self.data_path, 'parse', parse_name))
        parse = transforms.Resize(self.fine_width)(parse)
        parse = torch.from_numpy(np.array(parse)[None]).long()

        parse_13 = torch.FloatTensor(13, self.fine_height, self.fine_width).zero_()
        parse_13 = parse_13.scatter_(0, parse, 1.0)

        # image-parse-agnostic
        parse_ag_name = parse_name
        parse_ag = Image.open(osp.join(self.data_path, self.opt.parse_ag_mode, parse_ag_name))
        parse_ag = transforms.Resize(self.fine_width)(parse_ag)
        parse_ag = torch.from_numpy(np.array(parse_ag)[None]).long()

        parse_ag_13 = torch.FloatTensor(13, self.fine_height, self.fine_width).zero_()
        parse_ag_13 = parse_ag_13.scatter_(0, parse_ag, 1.0)
        
        # sk_vis
        sk_name = im_name
        sk_vis = Image.open(osp.join(self.data_path, 'sk-vis', sk_name))

        sk_vis = transforms.Resize(self.fine_width)(sk_vis)
        sk_vis = self.transform(sk_vis)

        # ck_vis
        if self.up == False:
            ck_name = c_name
            if self.data_list == 'demo_paired_pairs.txt':
                ck_vis = Image.open(osp.join(self.data_path, 'demo-paired-ck-vis', ck_name))
            else:
                ck_vis = Image.open(osp.join(self.data_path, 'paired-ck-vis', ck_name))
        
        else:
            ck_name = mix_name
            if self.data_list == 'demo_unpaired_pairs.txt':
                ck_vis = Image.open(osp.join(self.data_path, 'demo-unpaired-ck-vis', ck_name))
            else:
                ck_vis = Image.open(osp.join(self.data_path, 'unpaired-ck-vis', ck_name))        

        ck_vis = transforms.Resize(self.fine_width)(ck_vis)
        ck_vis = self.transform(ck_vis)

        result = {
            'im_name':  im_name,                          # image name
            'mix_name': mix_name,                         # mix_name
            'sk_vis': sk_vis,                             # for condition    
            'ck_vis': ck_vis,                             # for condition
            'parse_ag': parse_ag_13,                      # for condition
            'parse': parse_13,                            # ground truth
            }

        return result

    def __len__(self):
        return len(self.im_names)
    

class CPDataLoader(object):
    """
    Dataloader for Parse (Segmentation Map) Generation
    """
    def __init__(self, opt, dataset, shuffle=False):
        super(CPDataLoader, self).__init__()
        if shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=False,
                num_workers=opt.workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch



