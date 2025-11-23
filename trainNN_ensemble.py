import torch
import torch.nn as nn
from torch import optim
from nn_Model import nnModel
from evaluation import get_results
import numpy as np
from tqdm import tqdm
import json
from trainAE import process_adj_fea
import random

def set_seed(seed):
    """设置随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_nn(args,train_loader,device,input_dim,output_dim,go,test_loader,term):

    Epoch = 50
    
    # 集成学习：使用5个不同的随机种子训练模型
    seeds = [42, 123, 456, 789, 5959]  # 5个不同的随机种子
    num_ensembles = len(seeds)
    
    # 存储所有模型的预测结果
    all_ensemble_preds = []
    
    for ensemble_idx, seed in enumerate(seeds):
        print(f"\n{'='*50}")
        print(f"Training Ensemble Model {ensemble_idx + 1}/{num_ensembles} with seed {seed}")
        print(f"{'='*50}\n")
        
        # 设置随机种子
        set_seed(seed)
        
        model = nnModel(output_dim,dropout=0.2,device=device,args=args)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr/2)  #  0.0005
        bceloss = nn.BCELoss()
        weight_dict = {'lm_33':{},'lm_28':{},'lm_23':{}}

        for idx,e in enumerate(range(Epoch)):
            model.train()
            weight_dict['lm_33'][idx] = []
            weight_dict['lm_28'][idx] = []
            weight_dict['lm_23'][idx] = []
            for batch_idx,batch in enumerate(tqdm(train_loader,mininterval=0.5,desc=f'Ensemble {ensemble_idx+1} Training',leave=False,ncols=50)):
                optimizer.zero_grad()
                emb = batch[0].to(device)
                Y_label = batch[1].to(device)
                lm_33 = batch[2].to(device)
                lm_28 = batch[3].to(device)
                lm_23 = batch[4].to(device)
                id = batch[5].to(device)

                output,weight = model(emb.squeeze(),lm_33.squeeze(),lm_28.squeeze(),lm_23.squeeze())
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
                for batch_test_idx,batch_test in enumerate(tqdm(test_loader,mininterval=0.5,desc=f'Ensemble {ensemble_idx+1} Testing',leave=False,ncols=50)):
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

            if args.save_model:
                torch.save(model.state_dict(),'./data/'+args.species+'/trained_model/'+term+'/'+'seed'+str(seed)+'_Epoch_' + str(e + 1) + '-'+str(perf['all']['M-aupr'])+'.pkl')

            print('Epoch ' + str(e + 1) + '\tTrain loss:\t', loss_out.item(), '\tTest loss:\t',loss_test.item(), '\n\tM-AUPR:\t', perf['all']['M-aupr'], '\tm-AUPR:\t', perf['all']['m-aupr'],'\tF-max:\t', perf['all']['F-max'])

        
        # 保存该模型在测试集上的最终预测结果
        print(f"\nEnsemble Model {ensemble_idx + 1} training completed. Collecting final predictions...")
        model.eval()
        final_preds = torch.Tensor().to(device)
        final_labels = torch.Tensor().to(device)
        with torch.no_grad():
            for batch_test_idx,batch_test in enumerate(tqdm(test_loader,mininterval=0.5,desc=f'Collecting Ensemble {ensemble_idx+1} Predictions',leave=False,ncols=50)):
                label_test = batch_test[1].to(device)
                emb_test = batch_test[0].to(device)
                lm_33_test = batch_test[2].to(device)
                lm_28_test = batch_test[3].to(device)
                lm_23_test = batch_test[4].to(device)

                output_test,_ = model(emb_test.squeeze(),lm_33_test.squeeze(),lm_28_test.squeeze(),lm_23_test.squeeze())
                final_preds = torch.cat((final_preds, output_test), 0)
                final_labels = torch.cat((final_labels, label_test.squeeze()), 0)
        
        all_ensemble_preds.append(final_preds.cpu().numpy())
        
        # 保存单个模型的最终状态
        if args.save_model:
            torch.save(model.state_dict(),'./data/'+args.species+'/trained_model/'+term+'/'+'final_seed_'+str(seed)+'.pkl')
        
        # 清理GPU缓存
        del model
        torch.cuda.empty_cache()
    
    # 集成学习：平均所有模型的预测结果
    print(f"\n{'='*50}")
    print("Ensemble Learning: Averaging predictions from all models...")
    print(f"{'='*50}\n")
    
    ensemble_preds = np.mean(all_ensemble_preds, axis=0)  # 取平均
    ensemble_preds_tensor = torch.from_numpy(ensemble_preds).to(device)
    
    # 评估集成模型的性能
    perf_ensemble = get_results(go, final_labels.cpu().numpy(), ensemble_preds)
    
    print(f"\n{'='*50}")
    print("FINAL ENSEMBLE RESULTS:")
    print(f"{'='*50}")
    print(f'M-AUPR:\t{perf_ensemble["all"]["M-aupr"]:.6f}')
    print(f'm-AUPR:\t{perf_ensemble["all"]["m-aupr"]:.6f}')
    print(f'F-max:\t{perf_ensemble["all"]["F-max"]:.6f}')
    print(f"{'='*50}\n")
    
    # 保存集成预测结果
    np.save('./data/'+args.species+'/trained_model/'+term+'/ensemble_predictions.npy', ensemble_preds)
    
    # 保存每个单独模型的预测结果（可选）
    for i, preds in enumerate(all_ensemble_preds):
        np.save('./data/'+args.species+'/trained_model/'+term+f'/seed_{seeds[i]}_predictions.npy', preds)
    
    # 打印每个单独模型的性能（用于分析）
    print("\nIndividual Model Performance:")
    print(f"{'='*50}")
    for i, preds in enumerate(all_ensemble_preds):
        perf_single = get_results(go, final_labels.cpu().numpy(), preds)
        print(f"Seed {seeds[i]} - M-AUPR: {perf_single['all']['M-aupr']:.6f}, m-AUPR: {perf_single['all']['m-aupr']:.6f}, F-max: {perf_single['all']['F-max']:.6f}")
    print(f"{'='*50}\n")
