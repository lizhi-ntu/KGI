from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
import functools


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


class ConditionalGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(ConditionalGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        num_experts = 8
        self._routing_fn = _routing(in_features, num_experts, 0.0)
        self.weight = nn.Parameter(adj.unsqueeze(0).repeat(num_experts, 1, 1))
        nn.init.xavier_uniform_(self.weight)

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.M = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.M.data, gain=1.414)

        self.adj = adj

        #self.adj2 = nn.Parameter(torch.ones_like(adj))
        #nn.init.constant_(self.adj2, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        E0 = torch.eye(self.adj.size(1), dtype=torch.float).to(input.device)
        #E1 = torch.triu(torch.ones_like(self.adj), diagonal=1)
        #E2 = 1 - E1 - E0

        c = F.avg_pool2d(input, [self.adj.size(1), 1])
        r_w = self._routing_fn(c)
        cond_e = torch.sum(r_w[:, :, None, None] * self.weight, 1)

        # add modulation
        adj = self.adj[None, :].to(input.device) + cond_e

        # symmetry modulation
        adj = (adj.transpose(1, 2) + adj)/2

        # mix modulation
        output = torch.matmul(adj * E0, h0) + torch.matmul(adj * (1 - E0), self.M * h1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
