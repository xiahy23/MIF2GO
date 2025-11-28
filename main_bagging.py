import argparse
from input_data import load_data, load_labels
from trainAE import train_NoiseGAE
# 引入新的 Bagging 训练模块
from trainNN_bagging import train_nn 
import numpy as np
import pandas as pd
import os
import torch
from preprocessing import PFPDataset
from torch.utils.data import DataLoader
import warnings
import shutil
import time

def reshape(features):
    return np.hstack(features).reshape((len(features),len(features[0])))

def train(args):
    # 1. 自动替换 nn_Model 文件 (基于你之前的方案，确保用的是轻量化模型)
    # 建议使用 v2 (Attention Pooling) 配合 Bagging，效果最稳
    source_file = f'nn_Model_{args.model_select}.py'
    target_file = 'nn_Model.py'
    
    if os.path.exists(source_file):
        print(f"🔄 [Auto-Switch] Overwriting nn_Model.py with {source_file} ...")
        shutil.copy(source_file, target_file)
        time.sleep(1) 
    else:
        print(f"⚠️ Warning: {source_file} not found. Using current nn_Model.py")

    # 必须在替换文件后引入 nn_Model，虽然 train_nn 内部也会引
    # 这里主要是为了防止主进程缓存
    import importlib
    import trainNN_bagging
    importlib.reload(trainNN_bagging)

    print("loading features...")
    uniprot = pd.read_pickle(os.path.join(args.data_path, args.species, "features.pkl"))

    device = torch.device('cuda:'+args.device)
    
    # 加载图嵌入
    if 'embeddings.npy' not in os.listdir('./data/'+args.species+'/trained_emb_files/'):
        for graph in args.graphs:
            print(f"Processing {graph} data...")
            if graph == 'ppi':
                ppi_adj, ppi_features = load_data(graph, uniprot, args)
            else:
                ssn_adj, ssn_features = load_data(graph, uniprot, args)
        embeddings = train_NoiseGAE(ppi_features, ppi_adj,ssn_features,ssn_adj, args,device)
    else:
        print("Loading cached embeddings...")
        embeddings = np.load('./data/'+args.species+'/trained_emb_files/embeddings.npy')

    np.random.seed(5959)
    cc, mf, bp = load_labels(uniprot)

    # 数据划分
    num_test = int(np.floor(cc.shape[0] / 5.))
    num_train = cc.shape[0] - num_test

    if 'data_idx.txt' not in os.listdir('./data/'+args.species):
        all_idx = list(range(cc.shape[0]))
        np.random.shuffle(all_idx)
        with open('./data/'+args.species+'/data_idx.txt','a') as f:
            for idx in all_idx:
                f.write(str(idx)+'\n')
    else:
        all_idx = []
        with open('./data/'+args.species+'/data_idx.txt') as f:
            for line in f:
                all_idx.append(int(line.strip()))

    train_idx = all_idx[:num_train]
    test_idx = all_idx[num_train:(num_train + num_test)]

    # 加载 ESM 特征
    ESM_33 = np.load('./data/'+args.species+'/ESM-2_33.npy')
    ESM_28 = np.load('./data/' + args.species + '/ESM-2_28.npy')
    ESM_23 = np.load('./data/' + args.species + '/ESM-2_23.npy')

    Y_train_cc = cc[train_idx]; Y_test_cc = cc[test_idx]
    Y_train_bp = bp[train_idx]; Y_test_bp = bp[test_idx]
    Y_train_mf = mf[train_idx]; Y_test_mf = mf[test_idx]

    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]

    LM_train = [ESM_33[train_idx], ESM_28[train_idx], ESM_23[train_idx]]
    LM_test = [ESM_33[test_idx], ESM_28[test_idx], ESM_23[test_idx]]

    # 2. 创建数据集 (Dataset Objects)
    # 注意：这里不创建 DataLoader，因为 Bagging 需要在内部进行随机采样
    train_data_cc = PFPDataset(emb_X=X_train, data_Y=Y_train_cc, args=args, global_lm=LM_train)
    train_data_bp = PFPDataset(emb_X=X_train, data_Y=Y_train_bp, args=args, global_lm=LM_train)
    train_data_mf = PFPDataset(emb_X=X_train, data_Y=Y_train_mf, args=args, global_lm=LM_train)

    test_data_cc = PFPDataset(emb_X=X_test, data_Y=Y_test_cc, args=args, global_lm=LM_test)
    test_data_bp = PFPDataset(emb_X=X_test, data_Y=Y_test_bp, args=args, global_lm=LM_test)
    test_data_mf = PFPDataset(emb_X=X_test, data_Y=Y_test_mf, args=args, global_lm=LM_test)

    # 测试集的 Loader 是固定的，不需要 Bagging
    dataset_test_cc = DataLoader(test_data_cc, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    dataset_test_bp = DataLoader(test_data_bp, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    dataset_test_mf = DataLoader(test_data_mf, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    print("\n" + "="*60)
    print(f"🚀 Running Bagging Ensemble (Model: {args.model_select})")
    print("="*60)

    # 3. 训练 MF 任务
    print("\n>>> Processing MF Task...")
    trainNN_bagging.train_nn(
        args=args, device=device, 
        input_dim=embeddings.shape[1], output_dim=Y_train_mf.shape[1],
        train_dataset=train_data_mf, # 传入 Dataset
        test_loader=dataset_test_mf, # 传入 Loader
        go=mf, term='mf'
    )

    # 4. 训练 BP 任务
    print("\n>>> Processing BP Task...")
    trainNN_bagging.train_nn(
        args=args, device=device, 
        input_dim=embeddings.shape[1], output_dim=Y_train_bp.shape[1],
        train_dataset=train_data_bp, 
        test_loader=dataset_test_bp, 
        go=bp, term='bp'
    )

    # 5. 训练 CC 任务
    print("\n>>> Processing CC Task...")
    trainNN_bagging.train_nn(
        args=args, device=device, 
        input_dim=embeddings.shape[1], output_dim=Y_train_cc.shape[1],
        train_dataset=train_data_cc, 
        test_loader=dataset_test_cc, 
        go=cc, term='cc'
    )

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # 关键参数
    parser.add_argument('--model_select', type=str, default='v2', 
                       help="选择基模型架构: 'v2' (Attention Pooling - 推荐), 'v3' (Gated), 'original'")
    
    parser.add_argument('--ppi_attributes', type=int, default=5)
    parser.add_argument('--simi_attributes', type=int, default=5)
    parser.add_argument('--graphs', type=lambda s: [item for item in s.split(",")], default=['ppi','sequence_similarity'])
    parser.add_argument('--species', type=str, default="Human")
    parser.add_argument('--data_path', type=str, default="./data/")
    
    # Dropout 建议设大一点配合 Bagging
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate.") 
    
    parser.add_argument('--hidden1', type=int, default=800)
    parser.add_argument('--hidden2', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--thr_combined', type=float, default=0.4)
    parser.add_argument('--thr_evalue', type=float, default=1e-4)
    parser.add_argument('--noise_rate', type=float, default=0.6)
    parser.add_argument('--alpha', type=int, default=2)
    parser.add_argument('--eps', type=float, default=2.0)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--lambda_', type=float, default=0.4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_model', type=bool, default=False)

    args = parser.parse_args()
    train(args)