from __future__ import absolute_import

import torch.nn as nn
from functools import reduce

from models.gconv.vanilla_graph_conv import DecoupleVanillaGraphConv
from models.gconv.pre_agg_graph_conv import DecouplePreAggGraphConv
from models.gconv.post_agg_graph_conv import DecouplePostAggGraphConv
from models.gconv.conv_style_graph_conv import ConvStyleGraphConv
from models.gconv.no_sharing_graph_conv import NoSharingGraphConv
from models.gconv.sem_ch_graph_conv import SemCHGraphConv
from models.gconv.sem_graph_conv import SemGraphConv
from models.gconv.chebyshev_graph_conv import ChebyshevGraphConv
from models.gconv.modulated_gcn_conv import ModulatedGraphConv
from models.gconv.conditional_gcn_conv import ConditionalGraphConv

from models.graph_non_local import GraphNonLocal


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None, gcn_type=None, num_experts=0, reg_type=None):
        super(_GraphConv, self).__init__()

        if gcn_type == 'vanilla':
            self.gconv = DecoupleVanillaGraphConv(input_dim, output_dim, adj, decouple=False, num_experts=num_experts, reg_type=reg_type)
        elif gcn_type == 'dc_vanilla':
            self.gconv = DecoupleVanillaGraphConv(input_dim, output_dim, adj, num_experts=num_experts, reg_type=reg_type)
        elif gcn_type == 'preagg':
            self.gconv = DecouplePreAggGraphConv(input_dim, output_dim, adj, decouple=False, num_experts=num_experts, reg_type=reg_type)
        elif gcn_type == 'dc_preagg':
            self.gconv = DecouplePreAggGraphConv(input_dim, output_dim, adj, num_experts=num_experts, reg_type=reg_type)
        elif gcn_type == 'postagg':
            self.gconv = DecouplePostAggGraphConv(input_dim, output_dim, adj, decouple=False, num_experts=num_experts, reg_type=reg_type)
        elif gcn_type == 'dc_postagg':
            self.gconv = DecouplePostAggGraphConv(input_dim, output_dim, adj, num_experts=num_experts, reg_type=reg_type)
        elif gcn_type == 'convst':
            self.gconv = ConvStyleGraphConv(input_dim, output_dim, adj, num_experts=num_experts, reg_type=reg_type)
        elif gcn_type == 'nosharing':
            self.gconv = NoSharingGraphConv(input_dim, output_dim, adj, num_experts=num_experts, reg_type=reg_type)
        elif gcn_type == 'modulated':
            self.gconv = ModulatedGraphConv(input_dim, output_dim, adj, num_experts=num_experts, reg_type=reg_type)
        else:
            assert False, 'Invalid graph convolution module type'

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
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout, gcn_type=None, num_experts=0, reg_type=None):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout, gcn_type, num_experts, reg_type=reg_type)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout, gcn_type, num_experts, reg_type=reg_type)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(_GraphNonLocal, self).__init__()

        self.non_local = GraphNonLocal(hid_dim, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        out = self.non_local(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out


class GCN(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None, gcn_type=None, num_experts=6, reg_type=None):
        super(GCN, self).__init__()

        _gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout, gcn_type=gcn_type, num_experts=0, reg_type=reg_type)]
        _gconv_layers = []

        if nodes_group is None:
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout, gcn_type=gcn_type, num_experts=num_experts, reg_type=reg_type))
        else:
            group_size = len(nodes_group[0])
            assert group_size > 1

            grouped_order = list(reduce(lambda x, y: x + y, nodes_group))
            restored_order = [0] * len(grouped_order)
            for i in range(len(restored_order)):
                for j in range(len(grouped_order)):
                    if grouped_order[j] == i:
                        restored_order[i] = j
                        break

            _gconv_input.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout, gcn_type=gcn_type, num_experts=num_experts, reg_type=reg_type))
                _gconv_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)

        if gcn_type == 'vanilla':
            self.gconv_output = DecoupleVanillaGraphConv(hid_dim, coords_dim[1], adj, decouple=False, num_experts=num_experts, reg_type=reg_type)
        elif gcn_type == 'dc_vanilla':
            self.gconv_output = DecoupleVanillaGraphConv(hid_dim, coords_dim[1], adj, num_experts=num_experts, reg_type=reg_type)
        elif gcn_type == 'preagg':
            self.gconv_output = DecouplePreAggGraphConv(hid_dim, coords_dim[1], adj, decouple=False, num_experts=num_experts, reg_type=reg_type)
        elif gcn_type == 'dc_preagg':
            self.gconv_output = DecouplePreAggGraphConv(hid_dim, coords_dim[1], adj, num_experts=num_experts, reg_type=reg_type)
        elif gcn_type == 'postagg':
            self.gconv_output = DecouplePostAggGraphConv(hid_dim, coords_dim[1], adj, decouple=False, num_experts=num_experts, reg_type=reg_type)
        elif gcn_type == 'dc_postagg':
            self.gconv_output = DecouplePostAggGraphConv(hid_dim, coords_dim[1], adj, num_experts=num_experts, reg_type=reg_type)
        elif gcn_type == 'convst':
            self.gconv_output = ConvStyleGraphConv(hid_dim, coords_dim[1], adj, num_experts=num_experts, reg_type=reg_type)
        elif gcn_type == 'nosharing':
            self.gconv_output = NoSharingGraphConv(hid_dim, coords_dim[1], adj, num_experts=num_experts, reg_type=reg_type)
        elif gcn_type == 'modulated':
            self.gconv_output = ModulatedGraphConv(hid_dim, coords_dim[1], adj, num_experts=num_experts, reg_type=reg_type)
        else:
            assert False, 'Invalid graph convolution module type'

    def forward(self, x):
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        out = self.gconv_output(out)
        return out
