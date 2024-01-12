from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _routing(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.sigmoid(x)


class ConvStyleGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True, num_experts=0, reg_type='add', symmetric=True):
        super(ConvStyleGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conditional = num_experts > 0
        self.symmetric = symmetric
        self.reg_type = reg_type

        self.W = nn.Parameter(torch.zeros(size=(3, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

        if self.conditional:
            self._routing_fn = _routing(in_features, num_experts, 0.2)
            self.experts = nn.Parameter(adj.unsqueeze(0).repeat(num_experts, 1, 1))
            nn.init.xavier_uniform_(self.experts)

    def forward(self, input):
        adj = self.adj[None, :].to(input.device)

        if self.conditional:
            c = F.avg_pool2d(input, [self.adj.size(1), 1])
            r_w = self._routing_fn(c)
            cond_e = torch.sum(r_w[:, :, None, None] * self.experts, 1)
            if self.reg_type == 'add':
                adj = adj + cond_e
            elif self.reg_type == 'mul':
                adj = adj * cond_e
            elif self.reg_type == 'no_skeleton':
                adj = cond_e
            else:
                assert False, 'Invalid regulazation type'

            # symmetry modulation
            if self.symmetric:
                adj = (adj.transpose(1, 2) + adj) / 2

        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])
        h2 = torch.matmul(input, self.W[2])

        E0 = torch.eye(adj.size(1), dtype=torch.float).to(input.device)
        E1 = torch.triu(torch.ones_like(adj), diagonal=1)
        E2 = 1 - E1 - E0

        output = torch.matmul(adj * E0, h0) + torch.matmul(adj * E1, h1) + torch.matmul(adj * E2, h2)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'