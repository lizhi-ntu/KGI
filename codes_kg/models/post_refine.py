import torch
import torch.nn as nn

from torch.autograd import Variable


inter_channels = [128, 256]
fc_out = inter_channels[1]
fc_unit = 1024


class PostRefine(nn.Module):
    def __init__(self, in_channels, out_channels, n_joints):
        super().__init__()

        fc_in = out_channels * 2 * n_joints

        fc_out = in_channels * n_joints
        self.post_refine = nn.Sequential(
            nn.Linear(fc_in, fc_unit),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(fc_unit, fc_out),
            nn.Sigmoid()

        )

    def forward(self, x, x_1):
        """
        :param x:  N*V*3
        :param x_1: N*V*2
        :return:
        """
        # data normalization
        N, V,_ = x.size()
        x_in = torch.cat((x, x_1), -1)  #N*V*5
        x_in = x_in.view(N, -1)

        score = self.post_refine(x_in).view(N, V, 2)
        score_cm = Variable(torch.ones(score.size()), requires_grad=False).cuda() - score
        x_out = x.clone()
        x_out[:, :, :2] = score * x[:, :, :2] + score_cm * x_1[:, :, :2]

        return x_out
