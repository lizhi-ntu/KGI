#coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import json
import cv2
import os.path as osp
import numpy as np


class CPDataset(data.Dataset):
    """
    Dataset for Semantic TPS
    """
    def __init__(self, opt, datamode, data_list, up):
        super(CPDataset, self).__init__()
        self.opt = opt
        self.root = opt.dataroot
        self.up = up
        self.datamode = datamode
        self.data_list = data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.data_path = osp.join(opt.dataroot, datamode)

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
        self.label = json.load(open(osp.join(self.data_path, 'label.json')))

    def name(self):
        return "TPS Dataset"
    
    def __getitem__(self, index):
        im_name = self.im_names[index]
        if self.up == True:
            c_name = self.c_names[index]
            mix_name = '{}_{}'.format(im_name.split('.')[0], c_name)
        else:
            c_name = im_name
            mix_name = im_name
        
        # target image
        image = cv2.imread(osp.join(self.data_path, 'image', im_name), cv2.IMREAD_COLOR)
        image = np.array(transforms.Resize(self.fine_width)(Image.fromarray(image)))
        
        # cloth image
        cloth_name = c_name
        cloth = cv2.imread(osp.join(self.data_path, 'cloth', cloth_name), cv2.IMREAD_COLOR)
        cloth = np.array(transforms.Resize(self.fine_width)(Image.fromarray(cloth)))

        # ag_mask
        ag_mask_name = im_name.replace('.jpg', '.png')
        ag_mask = 255 - cv2.imread(osp.join(self.data_path, 'ag_mask', ag_mask_name), cv2.IMREAD_GRAYSCALE)
        ag_mask = np.array(transforms.Resize(self.fine_width)(Image.fromarray(ag_mask)))

        # skin_mask
        skin_mask_name = im_name.replace('.jpg', '.png')
        skin_mask = cv2.imread(osp.join(self.data_path, 'skin_mask', skin_mask_name), cv2.IMREAD_GRAYSCALE)
        skin_mask = np.array(transforms.Resize(self.fine_width)(Image.fromarray(skin_mask)))
        
        # parse 13
        if self.up == True:
            parse_name = mix_name.replace('.jpg', '.png')
            if self.data_list == 'demo_unpaired_pairs.txt':
                im_parse_pil_big = Image.open(osp.join(self.data_path, 'demo-unpaired-full-parse', parse_name))
            else:
                im_parse_pil_big = Image.open(osp.join(self.data_path, 'unpaired-full-parse', parse_name))
        else:
            parse_name = im_name.replace('.jpg', '.png')
            if self.data_list == 'demo_paired_pairs.txt':
                im_parse_pil_big = Image.open(osp.join(self.data_path, 'demo-paired-full-parse', parse_name))
            else:
                im_parse_pil_big = Image.open(osp.join(self.data_path, 'paired-full-parse', parse_name))
        
        im_parse_pil = transforms.Resize(self.fine_width)(im_parse_pil_big)
        parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()
        parse_13 = torch.FloatTensor(13, self.fine_height, self.fine_width).zero_()
        parse_13 = parse_13.scatter_(0, parse, 1.0)

        # cloth pos
        c_pos_name = c_name.replace('.jpg', '.json')
        c_pos = json.load(open(osp.join(self.data_path, 'cloth-landmark-json', c_pos_name)))
        key, _ = osp.splitext(self.im_names[index])
        if self.label[key] == 0:
            ck_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32] 
            c_pos = np.array(c_pos["long"])[ck_idx, :]
            
        if self.label[key] == 1:
            ck_idx = [1, 2, 3, 4, 5, 6, 6, 6, 6,  6,  7,  7,  7,  7,  7,  7,  8,  9, 10, 11, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14]
            c_pos = np.array(c_pos["vest"])[ck_idx, :]
        
        c_pos[:, 0] = c_pos[:, 0] / 3
        c_pos[:, 1] = c_pos[:, 1] / 4
        v_pos = torch.tensor(c_pos)

        # estimated cloth pos
        e_pos = 0
        if self.up == True:
            e_pos_name = mix_name.replace('.jpg', '.json')
            if self.data_list == 'demo_unpaired_pairs.txt':
                e_pos = json.load(open(osp.join(self.data_path, 'demo-unpaired-ck-point', e_pos_name)))
            else:
                e_pos = json.load(open(osp.join(self.data_path, 'unpaired-ck-point', e_pos_name)))
        else:
            e_pos_name = im_name.replace('.jpg', '.json')
            if self.data_list == 'demo_paired_pairs.txt':
                e_pos = json.load(open(osp.join(self.data_path, 'demo-paired-ck-point', e_pos_name)))
            else:
                e_pos = json.load(open(osp.join(self.data_path, 'paired-ck-point', e_pos_name)))
        e_pos = np.array(e_pos["keypoints"])
        e_pos = torch.tensor(e_pos)

        result = {
            'im_name': im_name,         # image name
            'mix_name': mix_name,       # mix name
            'image': image,             # target image
            'cloth': cloth,             # cloth image raw numpy array bgr
            'v_pos': v_pos,             # cloth keypoints position raw
            'e_pos': e_pos,             # estimated cloth keypoints position
            'ag_mask': ag_mask,
            'skin_mask': skin_mask,
            'parse_13': parse_13,
            }

        return result

    def __len__(self):
        return len(self.im_names)
    

class CPDataLoader(object):
    """
    Dataloader for Semantic TPS
    """
    def __init__(self, opt, dataset, shuffle=False):
        super(CPDataLoader, self).__init__()

        if shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
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