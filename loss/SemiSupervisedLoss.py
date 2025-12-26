import torch
import torch.nn as nn
import torch.nn.functional as F


"""
半监督损失
"""
class SemiSupervisedExpressionLoss(nn.Module):
    def __init__(self, sup_weight=1.0, unsup_weight=0.4, temp=0.1,frame=32):
        """
        参数说明：
            sup_weight: 监督损失权重（标记数据）
            unsup_weight: 无监督损失权重（未标记数据）
            temp: 时序一致性温度系数
        """
        super().__init__()
        self.sup_weight = sup_weight
        self.unsup_weight = unsup_weight
        self.temp = temp

        # 时序相邻帧的权重矩阵（可学习）
        self.register_buffer(
            "temporal_mask",
            torch.eye(frame) + torch.diag(torch.ones(frame-1), 1) + torch.diag(torch.ones(frame-1), -1)
        )

    def supervised_loss(self, pred, target):
        """支持动态序列长度的低内存消耗实现"""
        # 有效帧掩码 (B,T) = (8,32)
        valid_mask = (target.argmax(dim=-1) != 7)

        # 处理全无效批次（保留计算图）
        if valid_mask.sum() == 0:
            return pred.sum() * 0.0  # 零值梯度占位

        # 低内存索引（避免生成临时高维掩码）
        valid_indices = valid_mask.nonzero(as_tuple=True)
        valid_pred = pred[valid_indices[0], valid_indices[1]]  # (N_valid,7)
        valid_target = target[valid_indices[0], valid_indices[1], :7]  # (N_valid,)

        # 标签平滑（提升泛化性）
        return F.cross_entropy(valid_pred, valid_target, label_smoothing=0.1)



    def temporal_consistency_loss(self, pred):
        """基于时序平滑性的无监督损失"""
        # 计算帧间相似度（温度缩放余弦相似度）
        pred_norm = F.normalize(pred, p=2, dim=-1)  # (B, 32, 7)
        sim_matrix = torch.bmm(pred_norm, pred_norm.transpose(1, 2)) / self.temp  # (B, 32, 32)

        # 计算对比损失（强化相邻帧相似性）
        logits = sim_matrix - torch.max(sim_matrix, dim=2, keepdim=True)[0].detach()
        exp_logits = torch.exp(logits) * self.temporal_mask.unsqueeze(0)

        return -torch.log(exp_logits.sum(dim=2) / torch.exp(logits).sum(dim=2)).mean()

    def forward(self, pred, target=None):
        """
        前向计算：
            - 若有标签(target非None)：计算监督+无监督损失
            - 若无标签(仅pred)：仅计算无监督损失
        """
        total_loss = 0.0

        # 监督损失（当提供标签且权重>0时）
        if target is not None and self.sup_weight > 0:
            total_loss += self.sup_weight * self.supervised_loss(pred, target)

        # 无监督损失（当权重>0时）
        if self.unsup_weight > 0:
            total_loss += self.unsup_weight * self.temporal_consistency_loss(pred)

        return total_loss