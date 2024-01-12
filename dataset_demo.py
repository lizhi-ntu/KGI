import torch
import torch.utils.data as data
import json
import cv2
import os.path as osp
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

class CPDataset(data.Dataset):
    """
    Dataset for Demo with Pretrained Model
    """
    def __init__(self, opt, datamode, data_list, up=False):
        super(CPDataset, self).__init__()
        # base setting
        self.opt = opt
        self.up = up                                                        # if the person and cloth image unpaired
        self.root = opt.dataroot                                            # data root
        self.datamode = datamode                                            # train or test
        self.data_list = data_list                                          # data list
        self.fine_height = opt.fine_height                                  # image height
        self.fine_width = opt.fine_width                                    # image width
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
        self.label = json.load(open(osp.join(self.data_path, 'label.json'))) # denote whether the cloth has sleeves

    def name(self):
        return "CPDataset"
    
    def __getitem__(self, index):
        # target image
        im_name = self.im_names[index]
        image = cv2.imread(osp.join(self.data_path, 'image', im_name), cv2.IMREAD_COLOR)
        image = np.array(transforms.Resize(self.fine_width)(Image.fromarray(image)))
        
        if self.up == False:
            cloth_name = im_name
            mix_name = im_name.replace('.jpg', '.png')

        else:
            cloth_name = self.c_names[index]
            mix_name = '{}_{}'.format(im_name.split('.')[0], cloth_name).replace('.jpg', '.png')
                
        # cloth image
        cloth_name = cloth_name
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

        # skeleton pos
        s_pos_name = im_name.replace('.jpg', '_keypoints.json')
        s_pos = json.load(open(osp.join(self.data_path, 'openpose_json', s_pos_name)))["people"][0]["pose_keypoints_2d"]
        sk_idx = [0, 1, 2, 3, 4, 5, 6, 7, 9, 12]
        s_pos = np.resize(s_pos,(25,3))[sk_idx, 0:2]
        s_pos[:, 0] = s_pos[:, 0] / self.fine_width
        s_pos[:, 1] = s_pos[:, 1] / self.fine_height
        for l in range(10):
            if s_pos[l][0] == 0:
                if l in [0, 2, 5, 8, 9]:
                    s_pos[l, :] = s_pos[1, :]
                else:
                    s_pos[l, :] = s_pos[l-1, :]
        
        s_pos = torch.from_numpy(s_pos)

        # cloth pos
        c_pos_name = cloth_name.replace('.jpg', '.json')
        c_pos = json.load(open(osp.join(self.data_path, 'cloth-landmark-json', c_pos_name)))
        key, _ = osp.splitext(cloth_name)
        if self.label[key] == 0:
            ck_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32] 
            c_pos = np.array(c_pos["long"])[ck_idx, :]
            
        if self.label[key] == 1:
            ck_idx = [1, 2, 3, 4, 5, 6, 6, 6, 6,  6,  7,  7,  7,  7,  7,  7,  8,  9, 10, 11, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14]
            c_pos = np.array(c_pos["vest"])[ck_idx, :]
        
        c_pos[:, 0] = c_pos[:, 0] / 3
        c_pos[:, 1] = c_pos[:, 1] / 4
        v_pos = torch.tensor(c_pos)

        c_w = (c_pos[2][0] + c_pos[18][0]) / 2
        c_h = (c_pos[2][1] + c_pos[18][1]) / 2
        
        
        c_pos[:, 0] = c_pos[:, 0] - c_w
        c_pos[:, 1] = c_pos[:, 1] - c_h

        c_pos = torch.tensor(c_pos)
        
        # parsing image
        parse_name = im_name.replace('.jpg', '.png')
        parse = Image.open(osp.join(self.data_path, 'parse', parse_name))
        parse = transforms.Resize(self.fine_width)(parse)
        parse = torch.from_numpy(np.array(parse)[None]).long()

        parse_13 = torch.FloatTensor(13, self.fine_height, self.fine_width).zero_()
        parse_13 = parse_13.scatter_(0, parse, 1.0)
        
        # parsing agnostic image
        parse_ag_name = im_name.replace('.jpg', '.png')
        parse_ag = Image.open(osp.join(self.data_path, self.opt.parse_ag_mode, parse_ag_name))
        parse_ag = transforms.Resize(self.fine_width)(parse_ag)
        parse_ag = torch.from_numpy(np.array(parse_ag)[None]).long()

        parse_ag_13 = torch.FloatTensor(13, self.fine_height, self.fine_width).zero_()
        parse_ag_13 = parse_ag_13.scatter_(0, parse_ag, 1.0)
        
        result = {
            'im_name': im_name,         # image (person) name
            'cloth_name': cloth_name,   # cloth name
            'mix_name': mix_name,       # mix name  {im_name}_{cloth_name}
            'image': image,
            'cloth': cloth,
            'ag_mask': ag_mask,
            'skin_mask': skin_mask,
            'parse': parse_13,
            'parse_ag': parse_ag_13,
            's_pos': s_pos,             # skeleton keypoints posistion
            'c_pos': c_pos,             # centered cloth keypoints position
            'v_pos': v_pos,             # visualization (raw) cloth keypoints position
            }

        return result

    def __len__(self):
        return len(self.im_names)
    

class CPDataLoader(object):
    """
    Dataloader for Demo with Pretrained Model
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