import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from nn_Model import nnModel
from evaluation import get_results
import numpy as np
from tqdm import tqdm
import json
import random
import os

def set_seed(seed):
    """设置随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_nn(args, train_dataset, device, input_dim, output_dim, go, test_loader, term):
    """
    Bagging 训练函数
    Args:
        train_dataset: PFPDataset 对象 (注意：这里接收的是数据集，不是加载器)
        test_loader: 固定的测试集加载器
    """
    Epoch = 50
    
    # Bagging 设置：使用5个基学习器
    # 这里的种子用于控制 Bootstrap 采样的随机性
    seeds = [42, 123, 456, 789, 5959]  
    num_ensembles = len(seeds)
    
    # 存储所有模型的预测结果 [Num_Models, Num_Samples, Num_Classes]
    all_ensemble_preds = []
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting Bagging Ensemble for term: {term}")
    print(f"   - Base Learners: {num_ensembles}")
    print(f"   - Strategy: Bootstrap Sampling (replacement=True)")
    print(f"{'='*60}\n")

    for ensemble_idx, seed in enumerate(seeds):
        print(f"\n>>> Training Bagging Model {ensemble_idx + 1}/{num_ensembles} (Seed: {seed})")
        
        # 1. 设置随机种子 (控制模型初始化 和 数据采样)
        set_seed(seed)
        
        # 2. 构建 Bagging 数据加载器 (核心步骤)
        # 使用 RandomSampler 进行有放回采样 (replacement=True)
        # num_samples 保持与原数据集一致
        bagging_sampler = RandomSampler(
            train_dataset, 
            replacement=True, 
            num_samples=len(train_dataset)
        )
        
        # 注意：使用了 sampler 后，shuffle 必须为 False
        bagging_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            sampler=bagging_sampler, 
            drop_last=False,
            num_workers=args.num_workers
        )
        
        # 3. 初始化模型
        model = nnModel(output_dim, dropout=args.dropout, device=device, args=args)
        model = model.to(device)
        
        # 4. 优化器
        optimizer = optim.Adam(model.parameters(), lr=args.lr/2)
        bceloss = nn.BCELoss()

        # --- Training Loop ---
        for e in range(Epoch):
            model.train()
            epoch_loss = 0
            
            for batch in tqdm(bagging_loader, mininterval=1.0, desc=f'Ep {e+1} Train', leave=False, ncols=80):
                optimizer.zero_grad()
                emb = batch[0].to(device)
                Y_label = batch[1].to(device)
                lm_33 = batch[2].to(device)
                lm_28 = batch[3].to(device)
                lm_23 = batch[4].to(device)

                output, _ = model(emb.squeeze(), lm_33.squeeze(), lm_28.squeeze(), lm_23.squeeze())
                loss_out = bceloss(output, Y_label.squeeze())
                loss_out.backward()
                optimizer.step()
                
                epoch_loss += loss_out.item()

            # 简单打印一下进度 (每10个epoch或最后一个)
            if (e + 1) % 10 == 0 or (e + 1) == Epoch:
                print(f'   Epoch {e+1} | Avg Loss: {epoch_loss / len(bagging_loader):.4f}')

        # 5. 收集预测结果 (在固定测试集上)
        model.eval()
        final_preds = torch.Tensor().to(device)
        final_labels = torch.Tensor().to(device)
        
        with torch.no_grad():
            for batch_test in tqdm(test_loader, mininterval=0.5, desc=f'Inferencing Model {ensemble_idx+1}', leave=False, ncols=80):
                label_test = batch_test[1].to(device)
                emb_test = batch_test[0].to(device)
                lm_33_test = batch_test[2].to(device)
                lm_28_test = batch_test[3].to(device)
                lm_23_test = batch_test[4].to(device)

                output_test, _ = model(emb_test.squeeze(), lm_33_test.squeeze(), lm_28_test.squeeze(), lm_23_test.squeeze())
                final_preds = torch.cat((final_preds, output_test), 0)
                final_labels = torch.cat((final_labels, label_test.squeeze()), 0)
        
        # 评估单个模型性能
        current_preds_np = final_preds.cpu().numpy()
        all_ensemble_preds.append(current_preds_np)
        
        perf_single = get_results(go, final_labels.cpu().numpy(), current_preds_np)
        print(f"   ✅ Model {ensemble_idx+1} Result: F-max={perf_single['all']['F-max']:.4f}, M-AUPR={perf_single['all']['M-aupr']:.4f}")

        # 保存模型 (可选)
        if args.save_model:
            os.makedirs(f'./data/{args.species}/trained_model/{term}/bagging/', exist_ok=True)
            torch.save(model.state_dict(), f'./data/{args.species}/trained_model/{term}/bagging/model_{ensemble_idx+1}.pkl')
        
        # 释放显存
        del model
        torch.cuda.empty_cache()
    
    # 6. 集成聚合 (Averaging)
    print(f"\n{'='*60}")
    print("🤝 Aggregating Predictions (Bagging)...")
    print(f"{'='*60}")
    
    ensemble_preds = np.mean(all_ensemble_preds, axis=0)
    
    # 最终评估
    perf_ensemble = get_results(go, final_labels.cpu().numpy(), ensemble_preds)
    
    print(f"\n🏆 FINAL BAGGING RESULT [{term.upper()}]:")
    print(f"   M-AUPR: {perf_ensemble['all']['M-aupr']:.6f}")
    print(f"   m-AUPR: {perf_ensemble['all']['m-aupr']:.6f}")
    print(f"   F-max : {perf_ensemble['all']['F-max']:.6f}")
    print(f"{'='*60}\n")
    
    # 保存结果
    np.save(f'./data/{args.species}/trained_model/{term}/bagging_predictions.npy', ensemble_preds)