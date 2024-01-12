import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj
        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x

class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out

class GCN_2(nn.Module):
    def __init__(self, adj_c, adj_s, hid_dim, coords_dim=(2, 2, 2), num_layers=4, p_dropout=None):
        super(GCN_2, self).__init__()

        self.gconv_input = _GraphConv(adj_c, coords_dim[0], hid_dim, p_dropout=p_dropout)
        # 256x32x2 --> 256x32x160
        
        self.s_block1 = _GraphConv(adj_s, coords_dim[1], int(hid_dim/10*32), p_dropout=p_dropout)
        # 256x10x2 --> 256x10x512  --> 256x32x160
        self.c_block1 = nn.Sequential(
            _GraphConv(adj_c, hid_dim*2, hid_dim, p_dropout=p_dropout),
            _ResGraphConv(adj_c, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout)
            )
 
        self.s_block2 = _GraphConv(adj_s, int(hid_dim/10*32), int(hid_dim/10*32), p_dropout=p_dropout)
        self.c_block2 = nn.Sequential(
            _GraphConv(adj_c, hid_dim*2, hid_dim, p_dropout=p_dropout),
            _ResGraphConv(adj_c, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout)
            )        

        self.s_block3 = _GraphConv(adj_s, int(hid_dim/10*32), int(hid_dim/10*32), p_dropout=p_dropout)
        self.c_block3 = nn.Sequential(
            _GraphConv(adj_c, hid_dim*2, hid_dim, p_dropout=p_dropout),
            _ResGraphConv(adj_c, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout)
            ) 

        self.gconv_output = nn.Sequential(
            SemGraphConv(hid_dim, coords_dim[2], adj_c),
            nn.Sigmoid()
            )

    def forward(self, x_c, x_s):
        c_out = self.gconv_input(x_c)
        
        s_out = self.s_block1(x_s)
        c_out = self.c_block1(torch.cat([c_out, torch.transpose(s_out, 1, 2).view(s_out.shape[0], 32, 160)], dim=2))
        
        s_out = self.s_block2(s_out)
        c_out = self.c_block2(torch.cat([c_out, torch.transpose(s_out, 1, 2).view(s_out.shape[0], 32, 160)], dim=2))
        
        s_out = self.s_block3(s_out)
        c_out = self.c_block3(torch.cat([c_out, torch.transpose(s_out, 1, 2).view(s_out.shape[0], 32, 160)], dim=2))       
        
        c_out = self.gconv_output(c_out)
        return c_out