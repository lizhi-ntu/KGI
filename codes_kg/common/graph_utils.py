from __future__ import absolute_import

import torch
import numpy as np
import scipy.sparse as sp


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    
    #adj_mx = adj_mx * (1-torch.eye(adj_mx.shape[0])) + torch.eye(adj_mx.shape[0])

    return adj_mx


def adj_mx_from_skeleton(skeleton, graph_type, refine_type):
    num_joints = skeleton.num_joints()
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), skeleton.parents())))

    if graph_type == 'default':
        pass
    elif graph_type == 'double_chain':
        edges += [(2, 0), (5, 0), (8, 0), (7, 4), (6, 4), (7, 1), (3, 1), (9, 7), (11, 8), (14, 8), (12, 10), (15, 13), (13, 10)]
    elif graph_type == 'terminal_cycle':
        edges += [(3, 1), (6, 4), (12, 10), (15, 13)]
    elif graph_type == 'centralized':
        edges += [(7, 4), (7, 1), (10, 7), (13, 7)]
    elif graph_type == 'paired':
        edges += [(4, 1), (5, 2), (6, 3), (13, 10), (14, 11), (15,12)]
    else:
        assert False, 'Invalid graph kernel type'

    if refine_type == 'default':
        pass
    elif refine_type == 'self_weakening':
        edges += [edge for edge in edges]
    elif refine_type == 'self_reinforcement':
        edges += [(n, n) for n in range(num_joints)]
    else:
        assert False, 'Invalid refinement type'

    adj = adj_mx_from_edges(num_joints, edges, sparse=False)

    return adj
