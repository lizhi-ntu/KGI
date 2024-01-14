import os.path as osp
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


def load_data(data_root, data_list, batch_size, height=256, width=192, up=False):
    """
    Dataloader for Semantic Guided Inpainting
    """
    dataset = CPDataset(data_root=data_root, data_list=data_list, height=height, width=width, up=up)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
    while True:
        yield from loader

class CPDataset(Dataset):
    """
    Dataset for Semantic Guided Inpainting
    """
    def __init__(self, data_root, data_list, height=256, width=192, up=False):
        super().__init__()
        self.up = up
        self.data_root = data_root
        self.data_list = data_list
        self.height = height
        self.width = width
        self.data_path = '{}/{}'.format(data_root, 'test')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        # load data list
        im_names = []
        c_names = []
        with open('{}/{}'.format(self.data_root, self.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, idx):
        im_name = self.im_names[idx]
        if self.up == True:
            c_name = self.c_names[idx]
            mix_name = '{}_{}'.format(im_name.split('.')[0], c_name)
            if self.data_list == 'demo_unpaired_pairs.txt':
                mask_mid_name = 'mask_demo_unpaired_1024'
                image_mid_name = 'image_demo_unpaired_1024'
                parse_mid_name = 'demo-unpaired-full-parse'
            else:
                mask_mid_name = 'mask_unpaired_1024'
                image_mid_name = 'image_unpaired_1024'
                parse_mid_name = 'unpaired-full-parse'

        else:
            c_name = im_name
            mix_name = im_name
            if self.data_list == 'demo_paired_pairs.txt':
                mask_mid_name = 'mask_demo_paired_1024'
                image_mid_name = 'image_demo_paired_1024'
                parse_mid_name = 'demo-paired-full-parse'
            else:
                mask_mid_name = 'mask_paired_1024'
                image_mid_name = 'image_paired_1024'
                parse_mid_name = 'paired-full-parse'
        
        # incomplete image
        im_path = osp.join(self.data_path, image_mid_name, mix_name).replace('.jpg', '.png')
        im_pil_big = Image.open(im_path)
        im_pil = transforms.Resize(self.width)(im_pil_big)
        im = self.transform(im_pil)   
        
        out_dict = {}
        # estimated parse
        class_path = osp.join(self.data_path, parse_mid_name, mix_name).replace('.jpg', '.png')
        im_parse_pil_big = Image.open(class_path)
        im_parse_pil = transforms.Resize(self.width)(im_parse_pil_big)
        parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()
        
        parse_13 = torch.FloatTensor(13, self.height, self.width).zero_()
        parse_13 = parse_13.scatter_(0, parse, 1.0)
        
        # content keeping mask
        ag_mask_big = Image.open(osp.join(self.data_path, mask_mid_name, mix_name).replace('.jpg', '.png')).convert('RGB')
        ag_mask = transforms.Resize(self.width)(ag_mask_big)
        ag_mask = transforms.ToTensor()(ag_mask)

        out_dict['y'] = parse_13
        out_dict['gt_keep_mask'] = ag_mask
        out_dict['im_name'] = mix_name.replace('.jpg', '.png')
        
        return im, out_dict