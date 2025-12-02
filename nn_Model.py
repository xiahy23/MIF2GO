import torch.nn as nn
import torch
from torch.nn import functional as F
from GAE_model import double_NoiseGAE


# ==================== 版本4：轻量级门控融合 ====================
class GatedModalFusion(nn.Module):
    """替代 Transformer_scale_aggr 的轻量级门控融合"""
    def __init__(self, input_dim=400, dropout=0.2):
        super(GatedModalFusion, self).__init__()
        
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(input_dim)

    def forward(self, ppi, ssn, lm):
        concat_feat = torch.cat([ppi, ssn, lm], dim=-1)
        gates = self.gate_net(concat_feat)
        
        g_ppi = gates[:, 0].unsqueeze(1) * ppi
        g_ssn = gates[:, 1].unsqueeze(1) * ssn
        g_lm  = gates[:, 2].unsqueeze(1) * lm
        
        fused = g_ppi + g_ssn + g_lm
        return self.norm(fused)


# ==================== 原始 Transformer 融合层 ====================
class Transformer_scale_aggr(nn.Module):

    def __init__(self, dropout, device=None, heads=None):
        super(Transformer_scale_aggr, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(400, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.BatchNorm1d(400)
        self.norm2 = nn.BatchNorm1d(400)
        self.FFN = nn.Sequential(
            nn.Linear(400, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 400)
        )

        self.MLP_scale1 = nn.Sequential(
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.MLP_scale2 = nn.Sequential(
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.MLP_scale3 = nn.Sequential(
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, scale1, scale2, scale3):
        combined_feat = torch.cat([self.MLP_scale1(scale1), self.MLP_scale2(scale2), self.MLP_scale3(scale3)], dim=-1)
        output_ori = combined_feat.reshape(combined_feat.shape[0], 3, -1)
        output = self.self_attn(output_ori, output_ori, output_ori,
                                attn_mask=None,
                                key_padding_mask=None,
                                need_weights=False)[0]

        output = self.norm1(output_ori.reshape(-1, 400) + self.dropout(output.reshape(-1, 400)))

        dh = self.FFN(output)

        output = self.norm2(output + self.dropout(dh))

        return output.reshape(-1, 3, 400).sum(1)


# ==================== 主模型 ====================
class nnModel(nn.Module):

    def __init__(self, num_labels, dropout, device, args):
        super(nnModel, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.device = device
        self.args = args
        
        # 从 args 获取改进版本配置
        self.use_vib = getattr(args, 'use_vib', False)  # 版本1: VIB
        self.use_modal_dropout = getattr(args, 'use_modal_dropout', False)  # 版本2: Modal Dropout
        self.modal_dropout_rate = getattr(args, 'modal_dropout_rate', 0.3)
        self.use_shared_bottleneck = getattr(args, 'use_shared_bottleneck', False)  # 版本3: 共享瓶颈
        self.use_gated_fusion = getattr(args, 'use_gated_fusion', False)  # 版本4: 门控融合
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(400, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_labels),
        )

        # ==================== 版本3: 共享瓶颈层 ====================
        if self.use_shared_bottleneck:
            self.lm_bottleneck = nn.Sequential(
                nn.Linear(1280, 256),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.lm_fusion = nn.Linear(256 * 3, 400)
        else:
            # 原始 MLP_lm
            self.MLP_lm = nn.Sequential(
                nn.Linear(1280 * 3, 1280),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1280, 400),
            )

        # ==================== 版本4: 门控融合 vs 原始Transformer ====================
        if self.use_gated_fusion:
            self.scale_aggr = GatedModalFusion(400, dropout)
        else:
            self.scale_aggr = Transformer_scale_aggr(dropout, heads=args.heads)

        # ==================== 版本1: VIB 层 ====================
        if self.use_vib:
            self.vib_mu = nn.Linear(400, 400)
            self.vib_logvar = nn.Linear(400, 400)

        # 可学习的模态权重
        self.weight = torch.nn.init.constant_(nn.Parameter(torch.Tensor(3)), 1.0)

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def modal_dropout_mask(self, batch_size):
        """生成模态dropout掩码"""
        if self.training and self.use_modal_dropout:
            # 随机选择要 drop 的模态 (0, 1, 2 分别对应 PPI, SSN, LM)
            mask = torch.ones(3, device=self.device)
            if torch.rand(1).item() < self.modal_dropout_rate:
                drop_idx = torch.randint(0, 3, (1,)).item()
                mask[drop_idx] = 0.0
            return mask
        else:
            return torch.ones(3, device=self.device)

    def forward(self, emb, lm_33, lm_28, lm_23):
        weight = F.softmax(self.weight, dim=0)
        
        # ==================== 版本3: 共享瓶颈处理LM特征 ====================
        if self.use_shared_bottleneck:
            feat_33 = self.lm_bottleneck(lm_33)
            feat_28 = self.lm_bottleneck(lm_28)
            feat_23 = self.lm_bottleneck(lm_23)
            lm_feat = torch.cat([
                weight[0] * feat_33,
                weight[1] * feat_28,
                weight[2] * feat_23
            ], dim=-1)
            lm = self.lm_fusion(lm_feat)
        else:
            # 原始处理方式
            lm = torch.cat([weight[0] * lm_33, weight[1] * lm_28, weight[2] * lm_23], dim=-1)
            lm = self.MLP_lm(lm)

        # 分离 PPI 和 SSN 嵌入
        PPI_emb = emb[:, :400]
        SSN_emb = emb[:, 400:]

        # ==================== 版本2: 模态Dropout ====================
        if self.use_modal_dropout:
            mask = self.modal_dropout_mask(emb.shape[0])
            PPI_emb = PPI_emb * mask[0]
            SSN_emb = SSN_emb * mask[1]
            lm = lm * mask[2]

        # ==================== 融合层 ====================
        if self.use_gated_fusion:
            combined_feat = self.scale_aggr(PPI_emb, SSN_emb, lm)
        else:
            combined_feat = self.scale_aggr(PPI_emb, SSN_emb, lm)

        # ==================== 版本1: VIB ====================
        kl_loss = None
        if self.use_vib:
            mu = self.vib_mu(combined_feat)
            logvar = self.vib_logvar(combined_feat)
            combined_feat = self.reparameterize(mu, logvar)
            # 计算 KL 散度
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # 分类
        output = self.classifier(combined_feat)
        output = self.sigmoid(output)

        if self.use_vib:
            return output, weight, kl_loss
        else:
            return output, weight