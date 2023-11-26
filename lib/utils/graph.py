import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np

from .function import clones, l2norm
from .attention import weight_attention

import logging
logging.getLogger(__name__)


class RsGCNLayer(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1024):
        super(RsGCNLayer, self).__init__()

        self.out_dim = out_dim

        self.g = nn.Conv1d(in_channels=in_dim, out_channels=out_dim,
                           kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(nn.Conv1d(in_channels=out_dim, out_channels=in_dim,
                               kernel_size=1, stride=1, padding=0),
                               nn.BatchNorm1d(in_dim))
        nn.init.constant(self.W[1].weight, 0)
        nn.init.constant(self.W[1].bias, 0)

        self.theta = nn.Conv1d(in_channels=in_dim, out_channels=out_dim,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv1d(in_channels=in_dim, out_channels=out_dim,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, inp):
        nbatch, ninstance = inp.size(0), inp.size(-1)

        g_v = self.g(inp).view(nbatch, self.out_dim, -1).permute(0, 2, 1)
        theta_v = self.theta(inp).view(nbatch, self.out_dim, -1).permute(0, 2, 1)
        phi_v = self.phi(inp).view(nbatch, self.out_dim, -1)
        R = torch.matmul(theta_v, phi_v)
        R_div_C = R / R.size(-1)

        y = torch.matmul(R_div_C, g_v)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(nbatch, self.out_dim, ninstance)

        return self.W(y) + inp


class GraphReasonLayer(nn.Module):
    def __init__(self, dim, bias=True):
        super(GraphReasonLayer, self).__init__()

        self.graph_fcs = clones(nn.Linear(dim, dim, bias=bias), 3)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, inp):
        query, key = self.graph_fcs[0](inp), self.graph_fcs[1](inp)
        edge = self.softmax(torch.bmm(query, key.permute(0, 2, 1)))
        inp = torch.bmm(edge, inp)
        inp = self.relu(self.graph_fcs[2](inp))
        return inp


class GaussianKernel(nn.Module):
    def __init__(self, gk_type, n_kernels):
        super(GaussianKernel, self).__init__()
        self.gk_type = gk_type

        if 'img' in gk_type:
            # Parameters of the Gaussian kernels
            self.mean_rho = Parameter(torch.Tensor(n_kernels, 1))
            self.mean_theta = Parameter(torch.Tensor(n_kernels, 1))
            self.precision_rho = Parameter(torch.Tensor(n_kernels, 1))
            self.precision_theta = Parameter(torch.Tensor(n_kernels, 1))

            self.mean_theta.data.uniform_(-np.pi, np.pi)
            self.mean_rho.data.uniform_(0, 1.0)
            self.precision_theta.data.uniform_(0.0, 1.0)
            self.precision_rho.data.uniform_(0.0, 1.0)
        else:
            self.params = Parameter(torch.Tensor(n_kernels, 1))
            self.params.data.uniform_(-1.0, 1.0)

    def forward(self, features, depends, smooth=20.0):
        '''
        ## Inputs:
        - features (nbatch, nregion, emb_dim) or (nbatch, nword, emb_dim)
        - depends (nbatch, nregion, 4) or (npair, 2)
        ## Returns:
        - weights (nbatch, nregion, nneighbourhood, n_kernels) or
                  (nbatch, nword, nneighbourhood, n_kernels)
        '''

        nbatch, ninstance = features.size(0), features.size(1)
        adj_mtx = None

        if 'img' in self.gk_type:
            # Compute bbox centre (nbatch, nregion, 2)
            bb_size = (depends[:, :, 2:] - depends[:, :, :2])
            bb_centre = depends[:, :, :2] + 0.5 * bb_size

            # Compute cartesian coordinates (nbatch, nregion, nregion, 2)
            pseudo_coord = bb_centre.view(-1, ninstance, 1, 2) - bb_centre.view(-1, 1, ninstance, 2)

            # Conver to polar coordinates
            rho = torch.sqrt(pseudo_coord[:, :, :, 0] ** 2 + pseudo_coord[:, :, :, 1] ** 2)
            theta = torch.atan2(pseudo_coord[:, :, :, 0], pseudo_coord[:, :, :, 1])
            pseudo_coord = torch.cat((torch.unsqueeze(rho, -1), torch.unsqueeze(theta, -1)), dim=-1)
            pseudo_coord = pseudo_coord.to(features.device)

            # compute rho weights
            diff = (pseudo_coord[:, :, :, 0].contiguous().view(-1, 1) - self.mean_rho.view(1, -1)) ** 2
            weights_rho = torch.exp(-0.5 * diff /
                                    (1e-14 + self.precision_rho.view(1, -1) ** 2))
            # compute theta weights
            first_angle = torch.abs(pseudo_coord[:, :, :, 1].contiguous().view(-1, 1) - self.mean_theta.view(1, -1))
            second_angle = torch.abs(2 * np.pi - first_angle)
            weights_theta = torch.exp(-0.5 * (torch.min(first_angle, second_angle) ** 2)
                                      / (1e-14 + self.precision_theta.view(1, -1) ** 2))
            weights = weights_rho * weights_theta
            weights[(weights != weights).detach()] = 0
        else:
            _, weights = weight_attention(features, smooth)
            weights = weights.view(nbatch, ninstance, ninstance, -1)

            if 'sparse' in self.gk_type:
                # Build adjacency matrix for each text query
                adj = np.zeros((ninstance, ninstance), dtype=np.int)
                for i, pair in enumerate(depends):
                    if i == 0 or pair[0] >= ninstance or pair[1] >= ninstance:
                        continue
                    adj[pair[0], pair[1]] = 1
                    adj[pair[1], pair[0]] = 1
                adj = adj + np.eye(ninstance)

                adj_mtx = torch.from_numpy(adj).to(features.device).float()
                adj_mtx = adj_mtx.view(ninstance, ninstance).unsqueeze(0).unsqueeze(-1)
                # (nbatch, nword, nword, 1)
                weights = l2norm(adj_mtx * weights, dim=2)

            weights = weights.view(-1, 1) * self.params.view(1, -1)

        # normalise weights
        weights = weights.view(nbatch * ninstance, ninstance, -1)
        weights = weights / torch.sum(weights, dim=1, keepdim=True)
        weights = weights.view(nbatch, ninstance, ninstance, -1)

        return weights, adj_mtx


class GaussianKernelGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_kernels, bias=False):
        super(GaussianKernelGCNLayer, self).__init__()
        '''
        ## Variables:
        - in_dim: dimensionality of input features
        - out_dim: dimensionality of output features
        - n_kernels: number of Gaussian kernels to use
        - bias: whether to add a bias to convolutional kernels
        '''
        # Set parameters
        self.n_kernels = n_kernels
        self.out_dim = out_dim

        # Convolution filters weights
        self.conv_weights = nn.ModuleList([nn.Linear(
            in_dim, out_dim // n_kernels, bias=bias) for i in range(n_kernels)])

    def forward(self, neighbourhood_features, neighbourhood_weights):
        '''
        ## Inputs:
        - neighbourhood_features (nbatch, ninstance, nneighbourhood, in_dim)
        - neighbourhood_weights (nbatch, ninstance, nneighbourhood, n_kernels)
        ## Returns:
        - convolved_features (nbatch, ninstance, out_dim)
        '''

        # set parameters
        nbatch, ninstance, nneighbourhood, ndim = neighbourhood_features.size()

        # compute convolved features
        features = neighbourhood_features.view(
            nbatch * ninstance, nneighbourhood, -1)
        weights = neighbourhood_weights.view(
            nbatch * ninstance, nneighbourhood, self.n_kernels)

        convolved_features = self.convolution(features, weights)
        convolved_features = convolved_features.view(-1, ninstance, self.out_dim)

        return convolved_features

    def convolution(self, features, weights):
        '''
        ## Inputs:
        - features (nbatch*ninstance, compute_weights, in_dim)
        - weights (nbatch*ninstance, compute_weights, n_kernels)
        ## Returns:
        - convolved_features (nbatch*ninstance, out_dim)
        '''

        # patch operator
        weighted_features = torch.bmm(
            weights.transpose(1, 2), features)

        # convolutions
        weighted_features = [self.conv_weights[i](
            weighted_features[:, i]) for i in range(self.n_kernels)]
        convolved_features = torch.cat(
            [i.unsqueeze(1) for i in weighted_features], dim=1)
        convolved_features = convolved_features.view(-1, self.out_dim)

        return convolved_features


if __name__ == '__main__':
    GK_module = GaussianKernel('img', 8)
    TensorA = torch.Tensor(128, 36, 36, 2)
    out = GK_module(TensorA)

    print(out.size())