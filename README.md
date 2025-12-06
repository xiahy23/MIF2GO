# MIF2GO
Code for paper "MIF2GO: Annotating Protein Functions via Fusing Multiple Biological Modalities"
---

Dependencies
---

python == 3.7.16

pytorch == 1.13.1

PyG (torch-geometric) == 2.3.1

sklearn == 1.0.2

scipy == 1.7.3

numpy == 1.21.5

Data preparation (Password:1234)
---
For _human_, _fruit fly_, mouse, rat, S. cerevisiae, B. subtilis datasets:

1. The relevant data (~1.5G) can be available at the [Link](https://pan.baidu.com/s/11xFJtqn0ddIl4GUdrm3HvQ?pwd=1234).

2. Unzip the above file to the corresponding directory `./data/`.

3. If you want to train or test the model on different datasets, please modify the parameter settings in the code.

For CAFA dataset:

1. The relevant data (~4.5G) can be available at the [Link](https://pan.baidu.com/s/1EHGFid-cYMtOBcgi3nalbQ?pwd=1234).

2. Unzip the above file to the corresponding directory `./data/`.

Test
---
For _human_, _fruit fly_, mouse, rat, S. cerevisiae, B. subtilis datasets:

`python test.py` used to reproduct the performence recorded in the paper.

For CAFA dataset:

`python test_CAFA3.py` used to reproduct the performence recorded in the paper.

Train
---
For _human_, _fruit fly_, mouse, rat, S. cerevisiae, B. subtilis datasets:

`python main.py`

For CAFA dataset:

`python main_CAFA3.py`

P-value calculation
---
1. The relevant data (~1.5G) can be available at the [Link](https://pan.baidu.com/s/1HeTARs1y-VmJGCiGF17exw?pwd=1234).

2. Unzip the above file to the corresponding directory `./data/Human/`.

3. `python pvalue.py`

## Xia's experiments

## 4.1 PCA

- Switch to the `attempt_pca` branch.
- Run `python main_with_pca.py --use_pca --pca_components 256` ——降到256维

## 4.2 Loss func improvement

- Switch to the `attempt_loss` branch.
- Run `python main.py`（调参数在trainNN.py 第22行）

## 4.3 New models to relieve overfitting

- Switch to the `attempt_new_model` branch.
- Run `python main_with_pca.py --use_pca> logs/log_pcython main.py --use_vib --vib_beta 0.001 > logs/log_vib.txt 2>&1 & `
- Run `python main_with_pca.py --use_pca> logs/log_pcython main.py --use_modal_dropout --modal_dropout_rate 0.3 > logs/log_modal_dropout.txt 2>&1 &`
- Run `python main_with_pca.py --use_pca> logs/log_pcython main.py --use_shared_bottleneck > logs/log_shared_bottleneck.txt 2>&1 &`
- Run `python main_with_pca.py --use_pca> logs/log_pcython main.py --use_gated_fusion > logs/log_gated_fusion.txt 2>&1 &`
- 也可以组合使用
## 4.7 Emsemble Learning

- Switch to the `attempt_ensemble` branch.
- Run `python main_ensemble.py`
- Switch to the `attempt_bagging_stacking` branch.
- Run `python main_bagging.py`
- Run `python main_stacking.py`