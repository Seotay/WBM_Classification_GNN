import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class Patchifier(nn.Module):
    def __init__(self, in_channels=3, patch_emb=256):
        super().__init__()
        self.convs = nn.Sequential(
            nn. Conv2d(in_channels = in_channels, out_channels= 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 128 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 64 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, patch_emb, kernel_size=4, stride=2, padding=1), # 32 -> 16
            nn.BatchNorm2d(patch_emb),
            nn.ReLU(),
        )
         
    def forward(self, x):
        return self.convs(x)   # Patch: [B, patch_emb=256, 8, 8]

class DenseDilatedKnnGraph(nn.Module):
    def __init__(self, k=5, dilation=1):
        super().__init__()
        self.k = k
        self.dilation = dilation

    def forward(self, x):
        B, N, C = x.shape
        with torch.no_grad():
            dist = torch.cdist(x, x, p=2)                         # [B, N, N]
            knn_idx = dist.topk(k=self.k * self.dilation, dim=-1, largest=False).indices
            idx = knn_idx[..., ::self.dilation]                   # [B, N, k]

            src = torch.arange(N, device=x.device).view(1, N, 1).expand(B, N, self.k)  # [B,N,k]
            offset = (torch.arange(B, device=x.device) * N).view(B,1,1)
            src = (src + offset).reshape(-1).long()               # [B*N*k]
            dst = (idx  + offset).reshape(-1).long()              # [B*N*k]

            edge_index = torch.stack([src, dst], dim=0)           # [2, B*N*k]
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]

        return edge_index

class ViGBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.1):
        super().__init__()

        self.gcn = GCNConv(in_features, in_features)
        self.norm = nn.LayerNorm(in_features)
        self.drop = nn.Dropout(drop)
        self.act   = nn.GELU()

    def forward(self, x, edge_index, B, N):
        C = x.size(-1)
        shortcut = x

        x_flat = self.norm(x).reshape(B * N, C)
        out = self.gcn(x_flat, edge_index)              # [B*N, hidden]
        out = self.act(out)

        out = self.drop(out)
        out = out.reshape(B, N, -1)

        return out + shortcut

class ViGNN(nn.Module):
    def __init__(self, patch_emb=256, num_ViGBlocks=3, hidden_features=512,
                 k=5, dilation=1,  grid_size=8):
        super().__init__()

        self.patchifier = Patchifier(in_channels=3, patch_emb=patch_emb)
        self.graph_builder = DenseDilatedKnnGraph(k=k, dilation=dilation)

        self.grid_size = grid_size
        self.pos_embedding = nn.Parameter(torch.randn(1, self.grid_size*self.grid_size, patch_emb))

        self.blocks = nn.ModuleList([
            ViGBlock(in_features=patch_emb, hidden_features=hidden_features, drop=0.1)
            for _ in range(num_ViGBlocks)
        ])

        self.readout = nn.Sequential(
            nn.Conv2d(patch_emb, hidden_features, kernel_size=1, padding=0),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
            
    def forward(self, x):
        x = self.patchifier(x)                # [B, C, H, W] 
        B, C, H, W = x.shape
        N = H * W

        assert H == self.grid_size and W == self.grid_size, \
            f"Patch grid mismatch: got {H}x{W}, expected {self.grid_size}x{self.grid_size}"

        x_flat = x.view(B, C, N).permute(0, 2, 1)   # [B, N, C]
        x_flat = x_flat + self.pos_embedding[:, :N, :].to(dtype=x_flat.dtype)

        edge_index = self.graph_builder(x_flat)     # [2, E_total] with batch offset

        out = x_flat
        for block in self.blocks:
            out = block(out, edge_index, B, N)

        out = out.permute(0, 2, 1).view(B, C, H, W)
        out = self.readout(out).view(B, -1)
        return out



class ViG_Classifier(nn.Module):
    def __init__(self, patch_emb=256, num_blocks=3, hidden_features=512, n_classes=9, k=5, dilation=1,  grid_size=8):
        super().__init__()
        self.backbone = ViGNN(patch_emb=patch_emb, num_ViGBlocks=num_blocks,
                              hidden_features=hidden_features, k=k, dilation=dilation, grid_size=grid_size)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_features, 256),   
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.backbone(x)               # [B, patch_emb]
        logits = self.predictor(x)         # [B, n_classes]
        return x, logits