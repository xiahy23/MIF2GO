import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from nn_Model import nnModel
from evaluation import get_results
import numpy as np
from tqdm import tqdm
import os
import copy
import sys

# ---------------------------------------------------------
# Level-1 Meta-Learner (次级学习器)
# ---------------------------------------------------------
class MetaLearner(nn.Module):
    def __init__(self, num_labels):
        super(MetaLearner, self).__init__()
        self.classifier = nn.Linear(num_labels, num_labels)
        nn.init.eye_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.classifier(x))

def train_nn(args, train_dataset, device, input_dim, output_dim, go, test_loader, term):
    
    # 建议在终端运行前使用: export CUDA_LAUNCH_BLOCKING=1 以便精准定位错误
    
    # K-Fold 设置
    K = 5
    kf = KFold(n_splits=K, shuffle=True, random_state=5959)
    
    # OOF 预测矩阵初始化在 CPU 上 (安全模式)
    num_train_samples = len(train_dataset)
    oof_train_preds = torch.zeros((num_train_samples, output_dim)) 
    train_targets = torch.zeros((num_train_samples, output_dim))   
    
    test_preds_list = []
    
    print(f"\n{'='*60}")
    print(f"📚 Starting K-Fold Stacking (K={K}) for term: {term}")
    print(f"   - Mode: Robust (Clamping + NaN Checks)")
    print(f"{'='*60}\n")

    indices = np.arange(num_train_samples)
    
    # ====================================================
    # Phase 1: Level-0 Training (Base Learners)
    # ====================================================
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n>>> [Level-0] Training Fold {fold + 1}/{K}")
        
        fold_train_set = Subset(train_dataset, train_idx)
        fold_val_set   = Subset(train_dataset, val_idx)
        
        fold_train_loader = DataLoader(fold_train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        fold_val_loader   = DataLoader(fold_val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        model = nnModel(output_dim, dropout=args.dropout, device=device, args=args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr) 
        bceloss = nn.BCELoss()
        
        Fold_Epoch = 30 
        
        for e in range(Fold_Epoch):
            model.train()
            for batch_idx, batch in enumerate(tqdm(fold_train_loader, desc=f'Fold {fold+1} Ep {e+1}', leave=False, ncols=80)):
                optimizer.zero_grad()
                
                # [Safety 1] 检查数据完整性
                emb = batch[0].to(device)
                if torch.isnan(emb).any():
                    print(f"❌ Error: NaN detected in input embeddings at batch {batch_idx}!")
                    sys.exit(1)
                    
                Y_label = batch[1].to(device)
                # 确保标签是 float 类型且在 [0, 1] 之间
                if not ((Y_label >= 0) & (Y_label <= 1)).all():
                     print(f"❌ Error: Labels out of range [0, 1] at batch {batch_idx}!")
                     sys.exit(1)

                lm_33, lm_28, lm_23 = batch[2].to(device), batch[3].to(device), batch[4].to(device)
                
                output, _ = model(emb.squeeze(), lm_33.squeeze(), lm_28.squeeze(), lm_23.squeeze())
                
                # [Safety 2] 数值截断 (Clamping) - 防止 log(0) 导致 NaN/Inf
                # 这是解决 device-side assert 最有效的方法
                output = torch.clamp(output, min=1e-7, max=1.0-1e-7)
                
                loss = bceloss(output, Y_label.squeeze())
                loss.backward()
                optimizer.step()
        
        # --- 预测环节 (OOF) ---
        model.eval()
        fold_val_preds = []
        fold_val_targets = []
        
        with torch.no_grad():
            for batch in fold_val_loader:
                emb = batch[0].to(device)
                label = batch[1]
                lm_33, lm_28, lm_23 = batch[2].to(device), batch[3].to(device), batch[4].to(device)
                
                out, _ = model(emb.squeeze(), lm_33.squeeze(), lm_28.squeeze(), lm_23.squeeze())
                fold_val_preds.append(out.cpu())
                fold_val_targets.append(label.cpu())
            
            val_pred_tensor = torch.cat(fold_val_preds, dim=0)
            val_target_tensor = torch.cat(fold_val_targets, dim=0)
            
            # [Safety 3] 维度修复 (Shape Mismatch Fix)
            if val_target_tensor.dim() == 3 and val_target_tensor.shape[1] == 1:
                val_target_tensor = val_target_tensor.squeeze(1)
            
            # 对齐检查
            if val_pred_tensor.shape[0] != len(val_idx):
                min_len = min(val_pred_tensor.shape[0], len(val_idx))
                val_idx_trunc = val_idx[:min_len]
                val_pred_tensor = val_pred_tensor[:min_len]
                val_target_tensor = val_target_tensor[:min_len]
            else:
                val_idx_trunc = val_idx

            oof_train_preds[val_idx_trunc] = val_pred_tensor
            train_targets[val_idx_trunc]   = val_target_tensor

            # --- 预测环节 (Test Set) ---
            fold_test_preds = []
            for batch in test_loader:
                emb = batch[0].to(device)
                lm_33, lm_28, lm_23 = batch[2].to(device), batch[3].to(device), batch[4].to(device)
                
                out, _ = model(emb.squeeze(), lm_33.squeeze(), lm_28.squeeze(), lm_23.squeeze())
                fold_test_preds.append(out.cpu())
            
            test_preds_list.append(torch.cat(fold_test_preds, dim=0))
            
        print(f"    Fold {fold+1} Completed.")
        del model
        torch.cuda.empty_cache()

    # ====================================================
    # Phase 2: Level-1 Training (Meta Learner)
    # ====================================================
    print(f"\n>>> [Level-1] Training Meta-Learner...")
    
    X_meta_train = oof_train_preds.to(device)
    Y_meta_train = train_targets.to(device)
    X_meta_test = torch.stack(test_preds_list).mean(dim=0).to(device)
    
    meta_model = MetaLearner(output_dim).to(device)
    meta_optim = optim.Adam(meta_model.parameters(), lr=0.01)
    meta_loss_fn = nn.BCELoss()
    
    meta_model.train()
    for e in range(100): 
        meta_optim.zero_grad()
        meta_out = meta_model(X_meta_train)
        # [Safety 4] Meta-Learner 也要截断
        meta_out = torch.clamp(meta_out, min=1e-7, max=1.0-1e-7)
        
        loss = meta_loss_fn(meta_out, Y_meta_train)
        loss.backward()
        meta_optim.step()
        if (e+1) % 20 == 0:
            print(f"    Meta Train Ep {e+1}: Loss {loss.item():.6f}")
            
    # ====================================================
    # Phase 3: Final Evaluation
    # ====================================================
    print(f"\n>>> [Final] Evaluating...")
    meta_model.eval()
    
    test_labels_list = []
    for batch in test_loader:
        test_labels_list.append(batch[1])
    
    # [Safety 5] 测试集标签维度修复
    test_labels = torch.cat(test_labels_list).cpu()
    if test_labels.dim() == 3 and test_labels.shape[1] == 1:
        test_labels = test_labels.squeeze(1)
    test_labels_np = test_labels.numpy()
    
    with torch.no_grad():
        final_stacking_preds = meta_model(X_meta_test)
        final_preds_np = final_stacking_preds.cpu().numpy()
    
    perf = get_results(go, test_labels_np, final_preds_np)
    
    print(f"\n{'='*60}")
    print(f"🏆 FINAL STACKING RESULT [{term.upper()}]:")
    print(f"   M-AUPR: {perf['all']['M-aupr']:.6f}")
    print(f"   m-AUPR: {perf['all']['m-aupr']:.6f}")
    print(f"   F-max : {perf['all']['F-max']:.6f}")
    print(f"{'='*60}\n")
    
    result_dir = f'./data/{args.species}/trained_model/{term}'
    os.makedirs(result_dir, exist_ok=True)
    np.save(os.path.join(result_dir, 'stacking_predictions.npy'), final_preds_np)