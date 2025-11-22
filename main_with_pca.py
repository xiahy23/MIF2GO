"""
主训练脚本 - 集成PCA降维功能
基于main.py修改,添加ESM特征PCA降维预处理
"""

import argparse
from input_data import load_data,load_labels
from trainAE import train_NoiseGAE,train_GAE
from trainNN_pca import train_nn  # 使用支持PCA的训练函数
import numpy as np
import pandas as pd
import os
import torch
from preprocessing import PFPDataset
from torch.utils.data import DataLoader
import warnings
from scipy import sparse
from pca_utils import ESMFeaturePCA

def reshape(features):
    return np.hstack(features).reshape((len(features),len(features[0])))

def train(args):
    # load feature dataframe
    print("loading features...")
    uniprot = pd.read_pickle(os.path.join(args.data_path, args.species, "features.pkl"))

    device = torch.device('cuda:0')
    if 'embeddings.npy' not in os.listdir('./data/'+args.species+'/trained_emb_files/'):

        for graph in args.graphs:
            print("#############################")
            print(graph," data...")
            if graph == 'ppi':
                ppi_adj, ppi_features = load_data(graph, uniprot, args)
            else:
                ssn_adj, ssn_features = load_data(graph, uniprot, args)
        embeddings = train_NoiseGAE(ppi_features, ppi_adj,ssn_features,ssn_adj, args,device)
    else:
        embeddings = np.load('./data/'+args.species+'/trained_emb_files/embeddings.npy')

    np.random.seed(5959)

    cc, mf, bp = load_labels(uniprot)

    # split data into train and test
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

    # ======= 新增: 加载ESM特征 =======
    ESM_33 = np.load('./data/'+args.species+'/ESM-2_33.npy')
    ESM_28 = np.load('./data/' + args.species + '/ESM-2_28.npy')
    ESM_23 = np.load('./data/' + args.species + '/ESM-2_23.npy')

    # ======= 新增: PCA降维处理 =======
    if args.use_pca:
        print("\n" + "="*60)
        print(f"启用PCA降维: 1280维 -> {args.pca_components}维")
        print("="*60)
        
        # 创建PCA模型
        pca = ESMFeaturePCA(n_components=args.pca_components)
        
        # 检查是否已有保存的PCA模型
        pca_model_path = f'./data/{args.species}/trained_emb_files/pca_model_{args.pca_components}.pkl'
        
        if os.path.exists(pca_model_path) and not args.refit_pca:
            # 加载已有模型
            pca.load(pca_model_path)
        else:
            # 在训练集上拟合PCA
            print("在训练集上拟合PCA模型...")
            pca.fit(ESM_33[train_idx], ESM_28[train_idx], ESM_23[train_idx])
            # 保存PCA模型
            pca.save(pca_model_path)
        
        # 对训练集和测试集进行降维
        print("对训练集进行PCA降维...")
        ESM_33_train_reduced, ESM_28_train_reduced, ESM_23_train_reduced = pca.transform(
            ESM_33[train_idx], ESM_28[train_idx], ESM_23[train_idx]
        )
        
        print("对测试集进行PCA降维...")
        ESM_33_test_reduced, ESM_28_test_reduced, ESM_23_test_reduced = pca.transform(
            ESM_33[test_idx], ESM_28[test_idx], ESM_23[test_idx]
        )
        
        print(f"降维后特征形状: {ESM_33_train_reduced.shape}")
        print("="*60 + "\n")
        
        # 使用降维后的特征
        LM_train = [ESM_33_train_reduced, ESM_28_train_reduced, ESM_23_train_reduced]
        LM_test = [ESM_33_test_reduced, ESM_28_test_reduced, ESM_23_test_reduced]
    else:
        # 不使用PCA,使用原始特征
        print("\n未启用PCA降维,使用原始1280维特征\n")
        LM_train = [ESM_33[train_idx], ESM_28[train_idx], ESM_23[train_idx]]
        LM_test = [ESM_33[test_idx], ESM_28[test_idx], ESM_23[test_idx]]

    # ======= 准备数据 =======
    Y_train_cc = cc[train_idx]
    Y_train_bp = bp[train_idx]
    Y_train_mf = mf[train_idx]

    Y_test_cc = cc[test_idx]
    Y_test_bp = bp[test_idx]
    Y_test_mf = mf[test_idx]

    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]

    train_data_cc = PFPDataset(emb_X=X_train, data_Y=Y_train_cc,args=args,global_lm = LM_train)
    train_data_bp = PFPDataset(emb_X=X_train, data_Y=Y_train_bp,args=args,global_lm = LM_train)
    train_data_mf = PFPDataset(emb_X=X_train, data_Y=Y_train_mf,args=args,global_lm = LM_train)

    test_data_cc = PFPDataset(emb_X=X_test, data_Y=Y_test_cc,args=args,global_lm = LM_test)
    test_data_bp = PFPDataset(emb_X=X_test, data_Y=Y_test_bp,args=args,global_lm = LM_test)
    test_data_mf = PFPDataset(emb_X=X_test, data_Y=Y_test_mf,args=args,global_lm = LM_test)

    dataset_train_cc = DataLoader(train_data_cc, batch_size=args.batch_size, shuffle=True, drop_last=False,num_workers=args.num_workers)
    dataset_train_bp = DataLoader(train_data_bp, batch_size=args.batch_size, shuffle=True, drop_last=False,num_workers=args.num_workers)
    dataset_train_mf = DataLoader(train_data_mf, batch_size=args.batch_size, shuffle=True, drop_last=False,num_workers=args.num_workers)

    dataset_test_cc = DataLoader(test_data_cc, batch_size=args.batch_size, shuffle=False,  drop_last=False,num_workers=args.num_workers)
    dataset_test_bp = DataLoader(test_data_bp, batch_size=args.batch_size, shuffle=False,  drop_last=False,num_workers=args.num_workers)
    dataset_test_mf = DataLoader(test_data_mf, batch_size=args.batch_size, shuffle=False,  drop_last=False,num_workers=args.num_workers)

    print("Start running supervised model...")

    print("###################################")
    print('----------------------------------')
    print('MF')

    train_nn(args=args, device=device, input_dim=embeddings.shape[1], output_dim=Y_train_mf.shape[1],
             train_loader=dataset_train_mf,go=mf, test_loader=dataset_test_mf,term='mf')

    print("###################################")
    print('----------------------------------')
    print('BP')

    train_nn(args=args, device=device, input_dim=embeddings.shape[1], output_dim=Y_train_bp.shape[1],
             train_loader=dataset_train_bp, go=bp, test_loader=dataset_test_bp,term='bp')

    print("###################################")
    print('----------------------------------')
    print('CC')

    train_nn(args=args, device=device, input_dim=embeddings.shape[1], output_dim=Y_train_cc.shape[1],
             train_loader=dataset_train_cc, go=cc, test_loader=dataset_test_cc,term='cc')


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # global parameters
    parser.add_argument('--ppi_attributes', type=int, default=5, help="types of attributes used by ppi.(1 for pathway ablation, 2 for location ablation, 3 for domain ablation, 5 for using all feature)")
    parser.add_argument('--simi_attributes', type=int, default=5, help="types of attributes used by simi.(1 for pathway ablation, 2 for location ablation, 3 for domain ablation, 5 for using all feature)")
    parser.add_argument('--graphs', type=lambda s: [item for item in s.split(",")],
                        default=['ppi','sequence_similarity'], help="lists of graphs to use.")
    parser.add_argument('--species', type=str, default="Human", help="which species to use (Human/scerevisiae/rat/mouse/fly/ecoli).")
    parser.add_argument('--data_path', type=str, default="./data/", help="path storing data.")
    parser.add_argument('--hidden1', type=int, default=800, help="Number of units in hidden layer 1.")
    parser.add_argument('--hidden2', type=int, default=400, help="Number of units in hidden layer 2.")
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--epochs', type=int, default=160, help="Number of epochs to train ppi.")
    parser.add_argument('--thr_combined', type=float, default=0.4, help="threshold for ppi network.")
    parser.add_argument('--thr_evalue', type=float, default=0.0001, help="threshold for sequence similarity network.")
    parser.add_argument('--noise_rate', type=float, default=0.6, help="noise rate for feature.")
    parser.add_argument('--alpha', type=float, default=2, help="coefficient of reconstruction loss and contrastive learning loss.")
    parser.add_argument('--eps', type=float, default=2.0, help="margin of triplet loss.")
    parser.add_argument('--heads', type=int, default=4, help="number of heads in multi-head attention.")
    parser.add_argument('--lambda_', type=float, default=0.4, help="coefficient of penalty loss (asymmetric penalty).")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for DataLoader.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--save_model', action='store_true', default=False, help="Whether to save the trained model.")
    
    # ======= 新增: PCA相关参数 =======
    parser.add_argument('--use_pca', action='store_true', default=False, 
                        help="是否对ESM特征使用PCA降维")
    parser.add_argument('--pca_components', type=int, default=256, 
                        help="PCA降维后的维度 (推荐128或256)")
    parser.add_argument('--refit_pca', action='store_true', default=False,
                        help="是否强制重新拟合PCA模型")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("训练配置:")
    print(f"  物种: {args.species}")
    print(f"  使用PCA降维: {args.use_pca}")
    if args.use_pca:
        print(f"  PCA目标维度: {args.pca_components}")
    print("="*60 + "\n")
    
    train(args)
