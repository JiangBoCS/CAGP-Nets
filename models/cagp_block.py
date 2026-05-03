import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConstruction(nn.Module):
    """
    Constructs graph from feature map and performs message passing.
    Operates on P nodes each with C-dim features (channel vectors).
    """

    def __init__(self, node_dim, k=12):
        super().__init__()
        self.k = k
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(node_dim, node_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(self, nodes):
        """
        Args:
            nodes: (B, P, C) - P nodes each with C-dim feature
        Returns:
            updated_nodes: (B, P, C)
        """
        B, P, C = nodes.shape
        k = min(self.k, P - 1)

        # KNN via cosine similarity
        nodes_norm = F.normalize(nodes, dim=-1)
        sim = torch.bmm(nodes_norm, nodes_norm.transpose(1, 2))  # (B, P, P)
        sim.diagonal(dim1=1, dim2=2).fill_(-float('inf'))
        _, knn_idx = sim.topk(k, dim=-1)  # (B, P, k)

        # Gather neighbor features
        knn_idx_exp = knn_idx.unsqueeze(-1).expand(-1, -1, -1, C)
        neighbors = torch.gather(
            nodes.unsqueeze(1).expand(-1, P, -1, -1), 2, knn_idx_exp
        )  # (B, P, k, C)

        # Edge features: [q_i || q_j - q_i]
        center = nodes.unsqueeze(2).expand_as(neighbors)
        edge_input = torch.cat([center, neighbors - center], dim=-1)  # (B, P, k, 2C)

        # MLP + mean aggregation + LayerNorm
        edge_feat = self.edge_mlp(edge_input)  # (B, P, k, C)
        agg = edge_feat.mean(dim=2)  # (B, P, C)
        return self.layer_norm(agg)


class AdaptiveGraphPriors(nn.Module):
    """
    Generates adaptive graph prior via pixel-level and node-level embeddings.
    Resolution-independent: uses adaptive pooling for node-level compression.
    """

    def __init__(self, node_dim):
        super().__init__()
        # Pixel-level: compress channel axis -> 1
        self.proj_pixel = nn.Linear(node_dim, 1)
        # Node-level: adaptive pooling over node axis + linear
        self.proj_node = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
        )
        self.node_proj = nn.Linear(node_dim, node_dim)
        # Learnable per-channel re-weighting
        self.lambda_n = nn.Parameter(torch.ones(1, 1, node_dim))
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(self, F_g):
        """
        Args:
            F_g: (B, P, C)
        Returns:
            F_p: (B, P, C)
        """
        # Pixel-level embedding: (B, P, C) -> (B, P, 1)
        S_p = F.gelu(self.proj_pixel(F_g))  # (B, P, 1)
        # Node-level embedding: pool over P dim -> (B, 1, C)
        S_n = self.proj_node(F_g.transpose(1, 2)).transpose(1, 2)  # (B, 1, C)
        S_n = F.gelu(self.node_proj(S_n))  # (B, 1, C)
        # Broadcasting outer-sum
        S_m = S_p + S_n  # (B, P, C)
        # Element-wise re-weighting
        F_p = self.layer_norm(self.lambda_n * S_m)
        return F_p


class NodeClustering(nn.Module):
    """Differentiable node clustering with cosine similarity."""

    def __init__(self, node_dim, num_clusters=8, num_iters=3):
        super().__init__()
        self.num_clusters = num_clusters
        self.num_iters = num_iters
        self.proj = nn.Linear(node_dim, node_dim)
        self.refine = nn.Linear(node_dim, node_dim)

    def forward(self, F_p):
        """
        Args:
            F_p: (B, P, C)
        Returns:
            F_out: (B, P, C)
        """
        B, P, C = F_p.shape
        c = min(self.num_clusters, P)

        nodes = self.proj(F_p)

        # Initialize centers uniformly
        indices = torch.linspace(0, P - 1, c).long().to(F_p.device)
        centers = nodes[:, indices, :]  # (B, c, C)

        # Iterative clustering
        for _ in range(self.num_iters):
            nodes_norm = F.normalize(nodes, dim=-1)
            centers_norm = F.normalize(centers, dim=-1)
            sim = torch.bmm(nodes_norm, centers_norm.transpose(1, 2))  # (B, P, c)
            assignments = sim.argmax(dim=-1)  # (B, P)

            # Update centers
            new_centers = []
            for j in range(c):
                mask = (assignments == j).unsqueeze(-1).float()
                count = mask.sum(dim=1, keepdim=True).clamp(min=1)
                center_j = (nodes * mask).sum(dim=1, keepdim=True) / count
                new_centers.append(center_j)
            centers = torch.cat(new_centers, dim=1)

        # Soft assignment for differentiability
        nodes_norm = F.normalize(nodes, dim=-1)
        centers_norm = F.normalize(centers, dim=-1)
        sim = torch.bmm(nodes_norm, centers_norm.transpose(1, 2))
        weights = F.softmax(sim * 10.0, dim=-1)  # (B, P, c)
        clustered = torch.bmm(weights, centers)  # (B, P, C)

        return self.refine(clustered + F_p)


class CAGPBlock(nn.Module):
    """
    Clustered Adaptive Graph Priors Block.
    Lightweight design: operates on P nodes with C-dim features via adaptive avg pooling.
    """

    def __init__(self, channels=32, patch_size=8, k=12, num_clusters=8):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size

        self.graph = GraphConstruction(channels, k=k)
        self.adaptive_priors = AdaptiveGraphPriors(channels)
        self.node_clustering = NodeClustering(channels, num_clusters=num_clusters)
        self.fusion_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, F_a):
        """
        Args:
            F_a: (B, C, H, W)
        Returns:
            F_block: (B, C, H, W)
        """
        B, C, H, W = F_a.shape
        ps = self.patch_size

        # Ensure divisible
        pH, pW = H // ps, W // ps

        # Partition into patches and get per-patch features via mean pooling
        # (B, C, H, W) -> (B, C, pH, ps, pW, ps) -> mean over patch spatial dims -> (B, C, pH, pW)
        x = F_a[:, :, :pH*ps, :pW*ps]
        x = x.reshape(B, C, pH, ps, pW, ps)
        nodes = x.mean(dim=(3, 5))  # (B, C, pH, pW)
        nodes = nodes.reshape(B, C, -1).transpose(1, 2)  # (B, P, C)

        P = nodes.shape[1]

        # Graph construction and message passing
        graph_feat = self.graph(nodes)  # (B, P, C)

        # Adaptive graph priors
        F_p = self.adaptive_priors(graph_feat)  # (B, P, C)

        # Node clustering
        F_out = self.node_clustering(F_p)  # (B, P, C)

        # Reshape back to feature map via nearest-neighbor upsampling
        F_out = F_out.transpose(1, 2).reshape(B, C, pH, pW)  # (B, C, pH, pW)
        F_out_reshaped = F.interpolate(F_out, size=(H, W), mode='nearest')

        # Fusion: element-wise add + 1x1 conv
        F_block = self.fusion_conv(F_out_reshaped + F_a)
        return F_block
