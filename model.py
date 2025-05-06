import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np

class CLscETM(nn.Module):
    """
    持续学习的单细胞嵌入式主题模型 (CL-scETM)
    融合了BooVAE的持续学习能力和scETM的单细胞数据建模
    """
    
    def __init__(self, 
                 encoder,
                 decoder,
                 prior,
                 norm_cells=True,
                 normed_loss=True):
        """
        初始化CL-scETM模型
        
        参数:
            encoder: 编码器模型
            decoder: 解码器模型
            prior: 先验模型
            norm_cells: 是否标准化细胞输入
            normed_loss: 是否使用标准化损失
        """
        super(CLscETM, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        
        self.norm_cells = norm_cells
        self.normed_loss = normed_loss
        
        # 存储最佳主题和组件
        self.best_components = None
        
    def encode(self, x, batch_indices=None):
        """
        编码单细胞数据到潜变量
        
        参数:
            x: 输入数据 [batch_size, n_genes]
            batch_indices: 批次索引 [batch_size]
            
        返回:
            mu: 均值
            logsigma: 对数方差
            delta: 重参数化采样的潜变量
            theta: 标准化的主题混合
        """
        # 标准化输入
        if self.norm_cells:
            library_size = torch.sum(x, dim=1, keepdim=True)
            normed_x = x / (library_size + 1e-7)
        else:
            normed_x = x
            
        # 编码
        mu, logsigma = self.encoder(normed_x, batch_indices)
        
        # 重参数化采样
        if self.training:
            epsilon = torch.randn_like(mu)
            delta = mu + epsilon * torch.exp(0.5 * logsigma)
        else:
            delta = mu
            
        # 标准化主题混合（非负和为1）
        theta = F.softmax(delta, dim=1)
        
        return mu, logsigma, delta, theta
    
    def decode(self, theta, batch_indices=None):
        """
        解码主题混合到