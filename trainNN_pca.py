"""
训练函数 - 支持PCA降维后的可变ESM特征维度
基于trainNN.py修改
"""

import torch
import torch.nn as nn
from torch import optim
from nn_Model_pca import nnModel
from evaluation import get_results
from tqdm import tqdm


def train_nn(args, train_loader, device, input_dim, output_dim, go, test_loader, term):
    """
    训练神经网络模型
    
    Args:
        args: 参数对象
        train_loader: 训练数据加载器
        device: 设备
        input_dim: 输入维度(embedding维度)
        output_dim: 输出维度(标签数量)
        go: GO术语
        test_loader: 测试数据加载器
        term: 术语类型(mf/bp/cc)
    """

    Epoch = 50

    # 确定ESM特征维度
    esm_dim = args.pca_components if args.use_pca else 1280
    print(f"使用ESM特征维度: {esm_dim}")

    # 创建模型,传入esm_dim参数
    model = nnModel(output_dim, dropout=0.2, device=device, args=args, esm_dim=esm_dim)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr / 2)  # 0.0005

    bceloss = nn.BCELoss()

    weight_dict = {'lm_33': {}, 'lm_28': {}, 'lm_23': {}}

    for idx, e in enumerate(range(Epoch)):
        model.train()
        weight_dict['lm_33'][idx] = []
        weight_dict['lm_28'][idx] = []
        weight_dict['lm_23'][idx] = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, mininterval=0.5, desc='Training', leave=False, ncols=50)):
            optimizer.zero_grad()
            emb = batch[0].to(device)
            Y_label = batch[1].to(device)
            lm_33 = batch[2].to(device)
            lm_28 = batch[3].to(device)
            lm_23 = batch[4].to(device)

            output, weight = model(emb.squeeze(), lm_33.squeeze(), lm_28.squeeze(), lm_23.squeeze())
            loss_out = bceloss(output, Y_label.squeeze())
            loss_out.backward()
            optimizer.step()

            weight_dict['lm_33'][idx].append(float(weight[0].cpu()))
            weight_dict['lm_28'][idx].append(float(weight[1].cpu()))
            weight_dict['lm_23'][idx].append(float(weight[2].cpu()))

        model.eval()
        total_preds = torch.Tensor().to(device)
        total_labels = torch.Tensor().to(device)
        
        with torch.no_grad():
            for batch_test_idx, batch_test in enumerate(tqdm(test_loader, mininterval=0.5, desc='Testing', leave=False, ncols=50)):
                label_test = batch_test[1].to(device)
                emb_test = batch_test[0].to(device)
                lm_33_test = batch_test[2].to(device)
                lm_28_test = batch_test[3].to(device)
                lm_23_test = batch_test[4].to(device)

                output_test, _ = model(emb_test.squeeze(), lm_33_test.squeeze(), lm_28_test.squeeze(), lm_23_test.squeeze())
                total_preds = torch.cat((total_preds, output_test), 0)
                total_labels = torch.cat((total_labels, label_test.squeeze()), 0)

            loss_test = bceloss(total_preds, total_labels)

        perf = get_results(go, total_labels.cpu().numpy(), total_preds.cpu().numpy())

        # 修复: 使用 perf['all'] 访问性能指标
        print('Epoch ' + str(e + 1) + '\tTrain loss:\t', round(loss_out.item(), 4), 
              '\tTest loss:\t', round(loss_test.item(), 4), 
              '\n\tM-AUPR:\t', round(perf['all']['M-aupr'], 4), 
              '\tm-AUPR:\t', round(perf['all']['m-aupr'], 4),
              '\tF-max:\t', round(perf['all']['F-max'], 4))

    if args.save_model:
        import os
        model_save_path = f'./data/{args.species}/trained_model/{term}/'
        os.makedirs(model_save_path, exist_ok=True)
        
        if args.use_pca:
            model_file = f'model_pca{args.pca_components}.pkl'
        else:
            model_file = 'model.pkl'
            
        torch.save(model.state_dict(), os.path.join(model_save_path, model_file))
        print(f"模型已保存到: {os.path.join(model_save_path, model_file)}")
