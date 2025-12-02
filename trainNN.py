import torch
import torch.nn as nn
from torch import optim
from nn_Model import nnModel
from evaluation import get_results
import numpy as np
from tqdm import tqdm
import json
from trainAE import process_adj_fea


def train_nn(args, train_loader, device, input_dim, output_dim, go, test_loader, term):

    Epoch = 50

    model = nnModel(output_dim, dropout=0.2, device=device, args=args)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr / 2)

    bceloss = nn.BCELoss()

    # VIB 的 KL 权重 (beta-VAE 风格)
    vib_beta = getattr(args, 'vib_beta', 0.001)
    use_vib = getattr(args, 'use_vib', False)

    weight_dict = {'lm_33': {}, 'lm_28': {}, 'lm_23': {}}

    # 记录最佳模型结果
    best_fmax = 0.0
    best_epoch = 0
    best_perf = None

    for idx, e in enumerate(range(Epoch)):
        model.train()
        weight_dict['lm_33'][idx] = []
        weight_dict['lm_28'][idx] = []
        weight_dict['lm_23'][idx] = []
        
        total_train_loss = 0.0
        total_kl_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, mininterval=0.5, desc='Training', leave=False, ncols=50)):
            optimizer.zero_grad()
            emb = batch[0].to(device)
            Y_label = batch[1].to(device)
            lm_33 = batch[2].to(device)
            lm_28 = batch[3].to(device)
            lm_23 = batch[4].to(device)
            id = batch[5].to(device)

            # 前向传播
            if use_vib:
                output, weight, kl_loss = model(emb.squeeze(), lm_33.squeeze(), lm_28.squeeze(), lm_23.squeeze())
                bce_loss = bceloss(output, Y_label.squeeze())
                loss_out = bce_loss + vib_beta * kl_loss
                total_kl_loss += kl_loss.item()
            else:
                output, weight = model(emb.squeeze(), lm_33.squeeze(), lm_28.squeeze(), lm_23.squeeze())
                loss_out = bceloss(output, Y_label.squeeze())

            loss_out.backward()
            optimizer.step()

            total_train_loss += loss_out.item()

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

                if use_vib:
                    output_test, _, _ = model(emb_test.squeeze(), lm_33_test.squeeze(), lm_28_test.squeeze(), lm_23_test.squeeze())
                else:
                    output_test, _ = model(emb_test.squeeze(), lm_33_test.squeeze(), lm_28_test.squeeze(), lm_23_test.squeeze())
                
                total_preds = torch.cat((total_preds, output_test), 0)
                total_labels = torch.cat((total_labels, label_test.squeeze()), 0)

            loss_test = bceloss(total_preds, total_labels)

        perf = get_results(go, total_labels.cpu().numpy(), total_preds.cpu().numpy())

        # 更新最佳模型
        if perf['all']['F-max'] > best_fmax:
            best_fmax = perf['all']['F-max']
            best_epoch = e + 1
            best_perf = perf.copy()

        if args.save_model:
            torch.save(model.state_dict(), './data/' + args.species + '/trained_model/' + term + '/' + 'Epoch ' + str(e + 1) + '-' + str(perf['all']['M-aupr']) + '.pkl')

        # 打印信息
        log_msg = f"Epoch {e + 1}\tTrain loss:\t{total_train_loss / len(train_loader):.4f}\tTest loss:\t{loss_test.item():.4f}"
        if use_vib:
            log_msg += f"\tKL loss:\t{total_kl_loss / len(train_loader):.4f}"
        log_msg += f"\n\tM-AUPR:\t{perf['all']['M-aupr']:.4f}\tm-AUPR:\t{perf['all']['m-aupr']:.4f}\tF-max:\t{perf['all']['F-max']:.4f}"
        print(log_msg)

    # 输出最佳模型结果
    print("\n" + "=" * 60)
    print(f"Best Model ({term.upper()}) - Epoch {best_epoch}")
    print(f"  M-AUPR:\t{best_perf['all']['M-aupr']:.4f}")
    print(f"  m-AUPR:\t{best_perf['all']['m-aupr']:.4f}")
    print(f"  F-max:\t{best_perf['all']['F-max']:.4f}")
    print("=" * 60 + "\n")