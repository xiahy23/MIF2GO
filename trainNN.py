import torch
import torch.nn as nn
from torch import optim
from nn_Model import nnModel
from evaluation import get_results
import numpy as np
from tqdm import tqdm
import json
from trainAE import process_adj_fea
from asymmetric_loss import AsymmetricLoss

def train_nn(args,train_loader,device,input_dim,output_dim,go,test_loader,term):

    Epoch = 50

    model = nnModel(output_dim,dropout=0.2,device=device,args=args)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr/2)

    asl_loss = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    bceloss = nn.BCELoss()

    weight_dict = {'lm_33':{},'lm_28':{},'lm_23':{}}

    # 记录最佳模型性能
    best_perf = {
        'epoch': 0,
        'F-max': 0.0,
        'M-aupr': 0.0,
        'm-aupr': 0.0,
        'train_loss': 0.0,
        'test_loss': 0.0
    }

    for idx,e in enumerate(range(Epoch)):
        model.train()
        weight_dict['lm_33'][idx] = []
        weight_dict['lm_28'][idx] = []
        weight_dict['lm_23'][idx] = []
        for batch_idx,batch in enumerate(tqdm(train_loader,mininterval=0.5,desc='Training',leave=False,ncols=50)):
            optimizer.zero_grad()
            emb = batch[0].to(device)
            Y_label = batch[1].to(device)
            lm_33 = batch[2].to(device)
            lm_28 = batch[3].to(device)
            lm_23 = batch[4].to(device)
            id = batch[5].to(device)

            output,weight = model(emb.squeeze(),lm_33.squeeze(),lm_28.squeeze(),lm_23.squeeze())
            
            loss_out = asl_loss(output, Y_label.squeeze())
            loss_out.backward()
            optimizer.step()

            weight_dict['lm_33'][idx].append(float(weight[0].cpu()))
            weight_dict['lm_28'][idx].append(float(weight[1].cpu()))
            weight_dict['lm_23'][idx].append(float(weight[2].cpu()))

        model.eval()
        total_preds = torch.Tensor().to(device)
        total_labels = torch.Tensor().to(device)
        with torch.no_grad():
            for batch_test_idx,batch_test in enumerate(tqdm(test_loader,mininterval=0.5,desc='Testing',leave=False,ncols=50)):

                label_test = batch_test[1].to(device)
                emb_test = batch_test[0].to(device)
                lm_33_test = batch_test[2].to(device)
                lm_28_test = batch_test[3].to(device)
                lm_23_test = batch_test[4].to(device)

                output_test,_ = model(emb_test.squeeze(),lm_33_test.squeeze(),lm_28_test.squeeze(),lm_23_test.squeeze())
                total_preds = torch.cat((total_preds, output_test), 0)
                total_labels = torch.cat((total_labels, label_test.squeeze()), 0)

            loss_test = bceloss(total_preds,total_labels)

        perf = get_results(go, total_labels.cpu().numpy(), total_preds.cpu().numpy())

        # 更新最佳模型（基于F-max）
        if perf['all']['F-max'] > best_perf['F-max']:
            best_perf['epoch'] = e + 1
            best_perf['F-max'] = perf['all']['F-max']
            best_perf['M-aupr'] = perf['all']['M-aupr']
            best_perf['m-aupr'] = perf['all']['m-aupr']
            best_perf['train_loss'] = loss_out.item()
            best_perf['test_loss'] = loss_test.item()
            
            # 保存最佳模型
            if args.save_model:
                torch.save(model.state_dict(), './data/'+args.species+'/trained_model/'+term+'/best_model.pkl')

        if args.save_model:
            torch.save(model.state_dict(),'./data/'+args.species+'/trained_model/'+term+'/'+'Epoch ' + str(e + 1) + '-'+str(perf['all']['M-aupr'])+'.pkl')

        print('Epoch ' + str(e + 1) + '\tTrain loss:\t', loss_out.item(), '\tTest loss:\t',loss_test.item(), '\n\tM-AUPR:\t', perf['all']['M-aupr'], '\tm-AUPR:\t', perf['all']['m-aupr'],'\tF-max:\t', perf['all']['F-max'])

    # 打印最佳结果
    print('\n' + '='*60)
    print(f'Best Model at Epoch {best_perf["epoch"]}:')
    print(f'\tF-max:\t{best_perf["F-max"]:.6f}')
    print(f'\tM-AUPR:\t{best_perf["M-aupr"]:.6f}')
    print(f'\tm-AUPR:\t{best_perf["m-aupr"]:.6f}')
    print('='*60 + '\n')

    return best_perf