"""
神经网络模型 - 支持PCA降维后的可变ESM特征维度
基于nn_Model.py修改
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Transformer_scale_aggr(nn.Module):
    def __init__(self, dropout, d_model=400, heads=4):
        super(Transformer_scale_aggr, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, PPI_emb, SSN_emb, lm):
        combined_feat = torch.stack([PPI_emb, SSN_emb, lm], dim=1)
        combined_feat2 = self.multihead_attn(combined_feat, combined_feat, combined_feat, need_weights=False)[0]
        combined_feat = combined_feat + self.dropout1(combined_feat2)
        combined_feat = self.norm1(combined_feat)
        combined_feat2 = self.linear2(self.dropout(self.activation(self.linear1(combined_feat))))
        combined_feat = combined_feat + self.dropout2(combined_feat2)
        combined_feat = self.norm2(combined_feat)
        combined_feat = combined_feat.mean(dim=1)
        return combined_feat


class nnModel(nn.Module):
    """
    支持可变ESM特征维度的神经网络模型
    """

    def __init__(self, num_labels, dropout, device, args, esm_dim=1280):
        """
        Args:
            num_labels: 输出标签数量
            dropout: dropout率
            device: 设备
            args: 参数对象
            esm_dim: ESM特征维度 (默认1280, 使用PCA时为pca_components)
        """
        super(nnModel, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.device = device
        self.esm_dim = esm_dim

        self.classifier = nn.Sequential(
            nn.Linear(400, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_labels),
        )

        # 动态调整MLP_lm的输入维度
        self.MLP_lm = nn.Sequential(
            nn.Linear(esm_dim * 3, esm_dim),  # 输入: 3个ESM层的拼接
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(esm_dim, 400),  # 输出: 400维
        )

        self.scale_aggr = Transformer_scale_aggr(dropout, heads=args.heads)

        self.weight = torch.nn.init.constant_(nn.Parameter(torch.Tensor(3)), 1.0)

    def forward(self, emb, lm_33, lm_28, lm_23):
        weight = F.softmax(self.weight)

        lm = torch.cat([weight[0] * lm_33, weight[1] * lm_28, weight[2] * lm_23], dim=-1)

        lm = self.MLP_lm(lm)
        PPI_emb = emb[:, :400]
        SSN_emb = emb[:, 400:]
        lm = lm

        combined_feat = self.scale_aggr(PPI_emb, SSN_emb, lm)

        output = self.classifier(combined_feat)
        output = self.sigmoid(output)
        return output, weight
