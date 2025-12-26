import torch
import torch.nn as nn
import torch.nn.functional as F



"""
图对比损失
"""
class GraphContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, intra_graph_weight=0.5):
        super().__init__()
        self.temperature = temperature
        self.intra_graph_weight = intra_graph_weight

    def forward(self, node_embeddings, graph_labels):
        """
        Args:
            node_embeddings: (batch_size, num_nodes, feat_dim)
            graph_labels: (batch_size,) binary labels (0 or 1)
        """
        batch_size, num_nodes, feat_dim = node_embeddings.shape
        device = node_embeddings.device

        # 输入验证
        assert graph_labels.dim() == 1
        assert torch.all((graph_labels == 0) | (graph_labels == 1))

        total_loss = torch.tensor(0.0, device=device)
        valid_contrasts = 0

        #---- 图内对比 ----
        for i in range(batch_size):
            # 随机选择2个不同节点
            indices = torch.randperm(num_nodes)
            if len(indices) < 2:
                continue

            anchor = node_embeddings[i, indices[0]]  # (feat_dim,)
            positive = node_embeddings[i, indices[1]]  # (feat_dim,)

            # 计算正样本相似度
            pos_sim = F.cosine_similarity(
                anchor.unsqueeze(0),
                positive.unsqueeze(0),
                dim=1
            ) / self.temperature  # (1,)

            # 随机选择负样本 (来自其他图的节点)
            other_idx = torch.randint(0, batch_size, (1,)).item()
            while other_idx == i and batch_size > 1:
                other_idx = torch.randint(0, batch_size, (1,)).item()
            negative = node_embeddings[other_idx, torch.randint(0, num_nodes, (1,))]  # (1,feat_dim)

            # 计算负样本相似度
            neg_sim = F.cosine_similarity(
                anchor.unsqueeze(0),
                negative,
                dim=1
            ) / self.temperature  # (1,)

            # 构建logits和labels
            logits = torch.cat([pos_sim, neg_sim]).unsqueeze(0)  # (1,2)
            labels = torch.zeros(1, dtype=torch.long, device=device)

            total_loss += F.cross_entropy(logits, labels)
            valid_contrasts += 1


        if valid_contrasts == 0:
            return torch.tensor(0.0, device=device)

        return total_loss / valid_contrasts

