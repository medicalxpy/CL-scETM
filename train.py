import os
import time
import logging
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import scanpy as sc

# 设置日志
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

class CLscETMTrainer:
    """CL-scETM模型训练器"""
    
    def __init__(self, 
                 model, 
                 optimizer=None,
                 scheduler=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        初始化训练器
        
        参数:
            model: CL-scETM模型
            optimizer: 优化器，如果为None则创建默认的Adam优化器
            scheduler: 学习率调度器
            device: 计算设备
        """
        self.model = model
        self.device = device
        
        # 如果没有提供优化器，创建默认的Adam优化器
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer
            
        # 学习率调度器
        self.scheduler = scheduler
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_rec_loss': [],
            'train_kl_loss': [],
            'val_loss': [],
            'val_rec_loss': [],
            'val_kl_loss': [],
            'learning_rate': []
        }
        
        # 最佳模型性能
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
    def train_epoch(self, train_loader, beta=1.0, eps_gen=0.0):
        """
        训练一个epoch
        
        参数:
            train_loader: 训练数据加载器
            beta: KL散度权重
            eps_gen: 生成器正则化权重
            
        返回:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_kl_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            # 提取数据和批次信息
            if len(batch) == 2:
                x, batch_indices = batch
                batch_indices = batch_indices.to(self.device)
            else:
                x = batch[0]
                batch_indices = None
            
            x = x.to(self.device)
            
            # 前向传播和计算损失
            loss, rec_loss, kl_loss, _ = self.model.calculate_loss(
                x, batch_indices, beta=beta, eps_gen=eps_gen
            )
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 累积损失
            total_loss += loss.item()
            total_rec_loss += rec_loss.item()
            total_kl_loss += kl_loss.item()
            n_batches += 1
        
        # 计算平均损失
        avg_loss = total_loss / n_batches
        avg_rec_loss = total_rec_loss / n_batches
        avg_kl_loss = total_kl_loss / n_batches
        
        return avg_loss, avg_rec_loss, avg_kl_loss
    
    def validate(self, val_loader, beta=1.0, eps_gen=0.0):
        """
        验证模型
        
        参数:
            val_loader: 验证数据加载器
            beta: KL散度权重
            eps_gen: 生成器正则化权重
            
        返回:
            平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_kl_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 提取数据和批次信息
                if len(batch) == 2:
                    x, batch_indices = batch
                    batch_indices = batch_indices.to(self.device)
                else:
                    x = batch[0]
                    batch_indices = None
                
                x = x.to(self.device)
                
                # 计算损失
                loss, rec_loss, kl_loss, _ = self.model.calculate_loss(
                    x, batch_indices, beta=beta, eps_gen=eps_gen
                )
                
                # 累积损失
                total_loss += loss.item()
                total_rec_loss += rec_loss.item()
                total_kl_loss += kl_loss.item()
                n_batches += 1
        
        # 计算平均损失
        avg_loss = total_loss / n_batches
        avg_rec_loss = total_rec_loss / n_batches
        avg_kl_loss = total_kl_loss / n_batches
        
        return avg_loss, avg_rec_loss, avg_kl_loss
    
    def train(self, 
              train_loader, 
              val_loader=None, 
              n_epochs=100, 
              beta=1.0, 
              eps_gen=0.0,
              beta_warmup_epochs=0,
              save_dir=None,
              patience=10):
        """
        训练模型
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            n_epochs: 训练轮数
            beta: KL散度最终权重
            eps_gen: 生成器正则化权重
            beta_warmup_epochs: beta预热轮数
            save_dir: 模型保存目录
            patience: 早停耐心值
            
        返回:
            训练历史
        """
        _logger.info(f"开始训练，总轮数: {n_epochs}")
        
        # 初始化早停计数器
        patience_counter = 0
        
        # 创建保存目录
        if save_dir and not os.path.exists(save_dir):
            os.