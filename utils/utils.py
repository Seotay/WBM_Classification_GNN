import math
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import cv2

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resize_wafer_map(wmap, size=(64, 64)):
    return cv2.resize(wmap, size, interpolation=cv2.INTER_NEAREST)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    return: [grid_size*grid_size, embed_dim]
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
    # generate grid
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)

    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")  # 2D grid
    grid = torch.stack(grid, dim=0).reshape(2, -1)        # [2, N]

    # each position has (y,x)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    grid: [2, N]  (y, x)
    """
    assert embed_dim % 4 == 0
    half_dim = embed_dim // 2

    emb_y = get_1d_sincos_pos_embed_from_grid(half_dim, grid[0]) # y encoding
    emb_x = get_1d_sincos_pos_embed_from_grid(half_dim, grid[1]) # x encoding

    return torch.cat([emb_y, emb_x], dim=1)  # [N, embed_dim]


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: half dimension
    pos: [N]
    """
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))

    pos = pos.unsqueeze(1)     # [N,1]
    out = pos * omega          # [N, embed_dim/2]

    sin = torch.sin(out)
    cos = torch.cos(out)
    return torch.cat([sin, cos], dim=1)   # [N, embed_dim]



def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_inner = -2*torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def part_pairwise_distance(x, start_idx=0, end_idx=1):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = torch.sum(torch.mul(x_part, x_part), dim=-1, keepdim=True)
        x_inner = -2*torch.matmul(x_part, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square_part + x_inner + x_square.transpose(2, 1)


def xy_pairwise_distance(x, y):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        xy_inner = -2*torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)


def dense_knn_matrix(x, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        ### memory efficient implementation ###
        n_part = 10000
        if n_points > n_part:
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    dist += relative_pos[:, start_idx:end_idx]
                _, nn_idx_part = torch.topk(-dist, k=k)
                nn_idx_list += [nn_idx_part]
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            dist = pairwise_distance(x.detach())
            if relative_pos is not None:
                dist += relative_pos
            _, nn_idx = torch.topk(-dist, k=k) # b, n, k
        ######
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list

    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index


class DenseDilatedKnnGraph(nn.Module):
    def __init__(self, k=5, dilation=1):
        super().__init__()
        self.k = k
        self.dilation = dilation

    def forward(self, x):
        # x: [B, N, C]
        x = F.normalize(x, p=2.0, dim=2)
        B, N, C = x.shape

        with torch.no_grad():
            x = F.normalize(x, p=2, dim=2)          # [B,N,C]
            dist = torch.bmm(x, x.transpose(1,2))     # [B,N,N]
            knn_idx = dist.topk(k=self.k * self.dilation + 1, dim=-1, largest=True).indices  # [B,N,k*d+1]
            idx = knn_idx[..., 1::self.dilation]  

            src = torch.arange(N, device=x.device).view(1, N, 1).expand(B, N, self.k)
            offset = (torch.arange(B, device=x.device) * N).view(B, 1, 1)
            src = (src + offset).reshape(-1).long()
            dst = (idx + offset).reshape(-1).long()

            edge_index = torch.stack([src, dst], dim=0)  # [2, E]

        return edge_index

class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0, path="best_model.pt"):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_val_f1 = None
        self.early_stop = False
    def __call__(self, current_val_f1, model):
        if self.best_val_f1 is None:
            self.best_val_f1 = current_val_f1
            self.save_checkpoint(current_val_f1, model)

        elif current_val_f1 >= self.best_val_f1 + self.delta:
            self.save_checkpoint(current_val_f1, model)
            self.best_val_f1 = current_val_f1
            self.counter = 0

        else:
            self.counter += 1
            if self.verbose:
                print(f"\tEarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, current_val_f1, model):
        if self.verbose and self.best_val_f1 is not None:
            print(f"\tValidation F1 improved → Saving model: {self.path}")
        torch.save(model.state_dict(), self.path)

    def load_best_model(self, model):
        state_dict = torch.load(self.path, weights_only=True)
        model.load_state_dict(state_dict)

