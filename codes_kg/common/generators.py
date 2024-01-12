from __future__ import print_function, absolute_import

import numpy as np
import torch
from torch.utils.data import Dataset
from functools import reduce


class PoseGenerator(Dataset):
    def __init__(self, poses_3d, poses_2d, actions, cam):
        assert poses_3d is not None

        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d = np.concatenate(poses_2d)
        self._cam = np.concatenate(cam)

        self._actions = reduce(lambda x, y: x + y, actions)

        assert self._poses_3d.shape[0] == self._poses_2d.shape[0] and self._poses_3d.shape[0] == len(self._actions) and self._poses_3d.shape[0] == self._cam.shape[0]
        print('Generating {} poses...'.format(len(self._actions)))

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d = self._poses_2d[index]
        out_action = self._actions[index]
        out_cam = self._cam[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()
        out_cam = torch.from_numpy(out_cam).float()

        return out_pose_3d, out_pose_2d, out_action, out_cam

    def __len__(self):
        return len(self._actions)
