"""
PCA降维工具模块 - 用于对ESM-2特征进行降维处理
解决高维特征冗余导致的过拟合问题
"""

import numpy as np
import pickle
import os
from sklearn.decomposition import PCA
import torch


class ESMFeaturePCA:
    """
    ESM特征PCA降维器
    将1280维的ESM-2特征降维到指定维度(如128或256维)
    """
    
    def __init__(self, n_components=256, random_state=42):
        """
        初始化PCA降维器
        
        Args:
            n_components (int): 目标降维维度,默认256
            random_state (int): 随机种子,保证可重复性
        """
        self.n_components = n_components
        self.random_state = random_state
        self.pca_models = {
            'lm_33': None,
            'lm_28': None, 
            'lm_23': None
        }
        self.is_fitted = False
        
    def fit(self, lm_33_train, lm_28_train, lm_23_train):
        """
        在训练集上拟合PCA模型
        
        Args:
            lm_33_train: ESM-2 33层特征 [N_samples, 1280]
            lm_28_train: ESM-2 28层特征 [N_samples, 1280]
            lm_23_train: ESM-2 23层特征 [N_samples, 1280]
        """
        print(f"Fitting PCA on training data (降维到 {self.n_components} 维)...")
        
        # 确保输入是numpy数组
        if torch.is_tensor(lm_33_train):
            lm_33_train = lm_33_train.cpu().numpy()
        if torch.is_tensor(lm_28_train):
            lm_28_train = lm_28_train.cpu().numpy()
        if torch.is_tensor(lm_23_train):
            lm_23_train = lm_23_train.cpu().numpy()
            
        # 为每个ESM层创建独立的PCA模型
        self.pca_models['lm_33'] = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca_models['lm_28'] = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca_models['lm_23'] = PCA(n_components=self.n_components, random_state=self.random_state)
        
        # 拟合PCA模型
        self.pca_models['lm_33'].fit(lm_33_train)
        self.pca_models['lm_28'].fit(lm_28_train)
        self.pca_models['lm_23'].fit(lm_23_train)
        
        self.is_fitted = True
        
        # 打印解释方差比例
        print(f"PCA lm_33 解释方差比例: {self.pca_models['lm_33'].explained_variance_ratio_.sum():.4f}")
        print(f"PCA lm_28 解释方差比例: {self.pca_models['lm_28'].explained_variance_ratio_.sum():.4f}")
        print(f"PCA lm_23 解释方差比例: {self.pca_models['lm_23'].explained_variance_ratio_.sum():.4f}")
        
    def transform(self, lm_33, lm_28, lm_23):
        """
        对特征进行PCA降维
        
        Args:
            lm_33: ESM-2 33层特征 [N_samples, 1280]
            lm_28: ESM-2 28层特征 [N_samples, 1280]
            lm_23: ESM-2 23层特征 [N_samples, 1280]
            
        Returns:
            降维后的特征 [N_samples, n_components]
        """
        if not self.is_fitted:
            raise ValueError("PCA模型未拟合,请先调用fit()方法!")
            
        # 确保输入是numpy数组
        is_torch = torch.is_tensor(lm_33)
        if is_torch:
            lm_33 = lm_33.cpu().numpy()
            lm_28 = lm_28.cpu().numpy()
            lm_23 = lm_23.cpu().numpy()
            
        # 进行降维
        lm_33_reduced = self.pca_models['lm_33'].transform(lm_33)
        lm_28_reduced = self.pca_models['lm_28'].transform(lm_28)
        lm_23_reduced = self.pca_models['lm_23'].transform(lm_23)
        
        # 如果输入是tensor,转回tensor
        if is_torch:
            lm_33_reduced = torch.tensor(lm_33_reduced, dtype=torch.float32)
            lm_28_reduced = torch.tensor(lm_28_reduced, dtype=torch.float32)
            lm_23_reduced = torch.tensor(lm_23_reduced, dtype=torch.float32)
            
        return lm_33_reduced, lm_28_reduced, lm_23_reduced
    
    def fit_transform(self, lm_33_train, lm_28_train, lm_23_train):
        """
        拟合并转换训练数据
        
        Args:
            lm_33_train: ESM-2 33层特征 [N_samples, 1280]
            lm_28_train: ESM-2 28层特征 [N_samples, 1280]
            lm_23_train: ESM-2 23层特征 [N_samples, 1280]
            
        Returns:
            降维后的特征 [N_samples, n_components]
        """
        self.fit(lm_33_train, lm_28_train, lm_23_train)
        return self.transform(lm_33_train, lm_28_train, lm_23_train)
    
    def save(self, save_path):
        """
        保存PCA模型到文件
        
        Args:
            save_path (str): 保存路径
        """
        if not self.is_fitted:
            raise ValueError("PCA模型未拟合,无法保存!")
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'pca_models': self.pca_models,
                'n_components': self.n_components,
                'random_state': self.random_state,
                'is_fitted': self.is_fitted
            }, f)
        print(f"PCA模型已保存到: {save_path}")
        
    def load(self, load_path):
        """
        从文件加载PCA模型
        
        Args:
            load_path (str): 加载路径
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"PCA模型文件不存在: {load_path}")
            
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
            
        self.pca_models = data['pca_models']
        self.n_components = data['n_components']
        self.random_state = data['random_state']
        self.is_fitted = data['is_fitted']
        print(f"PCA模型已从 {load_path} 加载")
        print(f"降维维度: {self.n_components}")


if __name__ == "__main__":
    # 测试代码
    print("PCA降维工具模块加载成功!")
    print("使用示例:")
    print("  from pca_utils import ESMFeaturePCA")
    print("  pca = ESMFeaturePCA(n_components=256)")
    print("  pca.fit(lm_33_train, lm_28_train, lm_23_train)")
    print("  lm_33_reduced, lm_28_reduced, lm_23_reduced = pca.transform(lm_33, lm_28, lm_23)")
