import os.path as osp
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


def load_data(data_root, data_list, batch_size, height=256, width=192):
    """
    Dataloader for Semantic Diffusion Model Training
    """
    dataset = CPDataset(data_root, data_list, height=height, width=width)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    while True:
        yield from loader


class CPDataset(Dataset):
    """
    Dataset for Semantic Diffusion Model Training
    """
    def __init__(self, data_root, data_list, height=256, width=192):
        super().__init__()
        self.data_root = data_root
        self.data_list = data_list
        self.height = height
        self.width = width
        self.data_path = '{}/{}'.format(data_root, 'train')
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
        im_path = osp.join(self.data_path, 'image', im_name)
        im_pil_big = Image.open(im_path)
        im_pil = transforms.Resize(self.width)(im_pil_big)
        im = self.transform(im_pil)   
        
        out_dict = {}
        class_path = osp.join(self.data_path, 'parse', im_name).replace('.jpg', '.png')
        im_parse_pil_big = Image.open(class_path)
        im_parse_pil = transforms.Resize(self.width)(im_parse_pil_big)
        parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()
        parse_13 = torch.FloatTensor(13, self.height, self.width).zero_()
        parse_13 = parse_13.scatter_(0, parse, 1.0)

        out_dict['y'] = parse_13
        
        return im, out_dict
