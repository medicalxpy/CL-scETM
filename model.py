import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import os
from typing import Union, Tuple, Dict, List, Optional
import logging

_logger = logging.getLogger(__name__)

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
                 normed_loss=True,
                 normalize_beta=False,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        初始化CL-scETM模型
        
        参数:
            encoder: 编码器模型
            decoder: 解码器模型
            prior: 先验模型
            norm_cells: 是否标准化细胞输入
            normed_loss: 是否使用标准化损失
            normalize_beta: 是否标准化beta矩阵
            device: 计算设备
        """
        super(CLscETM, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        
        self.norm_cells = norm_cells
        self.normed_loss = normed_loss
        self.normalize_beta = normalize_beta
        self.device = device
        
        # 存储最佳主题和组件
        self.best_components = None
        
        # 当前任务ID
        self.current_task_id = 0
        
        # 模型移至指定设备
        self.to(device)
        
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
        解码主题混合到基因表达
        
        参数:
            theta: 主题混合 [batch_size, n_topics]
            batch_indices: 批次索引 [batch_size]
            
        返回:
            recon_x: 重构的基因表达 [batch_size, n_genes]
            recon_log: 重构的对数基因表达 [batch_size, n_genes]
        """
        # 通过解码器重构基因表达
        recon_x = self.decoder(theta, batch_indices, self.normalize_beta)
        
        # 计算对数表达
        recon_log = torch.log(recon_x + 1e-30)
        
        return recon_x, recon_log
    
    def forward(self, x, batch_indices=None):
        """
        前向传播
        
        参数:
            x: 输入数据 [batch_size, n_genes]
            batch_indices: 批次索引 [batch_size]
            
        返回:
            recon_x: 重构的基因表达
            recon_log: 重构的对数基因表达
            mu: 均值
            logsigma: 对数方差
            delta: 重参数化采样的潜变量
            theta: 标准化的主题混合
        """
        # 编码
        mu, logsigma, delta, theta = self.encode(x, batch_indices)
        
        # 解码
        recon_x, recon_log = self.decode(theta, batch_indices)
        
        return recon_x, recon_log, mu, logsigma, delta, theta
    
    def calculate_loss(self, x, batch_indices=None, beta=1.0, eps_gen=0.0):
        """
        计算模型损失
        
        参数:
            x: 输入数据 [batch_size, n_genes]
            batch_indices: 批次索引 [batch_size]
            beta: KL散度权重
            eps_gen: 生成器正则化权重
            
        返回:
            loss: 总损失
            rec_loss: 重构损失
            kl_loss: KL散度损失
            reg_loss: 正则化损失
        """
        # 前向传播
        recon_x, recon_log, mu, logsigma, delta, theta = self.forward(x, batch_indices)
        
        # 计算重构损失
        if self.normed_loss:
            # 使用标准化数据计算损失
            library_size = torch.sum(x, dim=1, keepdim=True)
            normed_x = x / (library_size + 1e-7)
            rec_loss = -torch.sum(normed_x * recon_log, dim=1).mean()
        else:
            # 使用原始数据计算损失
            rec_loss = -torch.sum(x * recon_log, dim=1).mean()
        
        # 计算KL散度
        prior_log_density = self.prior(delta)
        log_q_z = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp(), dim=1)
        kl_loss = (log_q_z - prior_log_density).mean()
        
        # 计算总损失
        loss = rec_loss + beta * kl_loss
        
        # 添加正则化项（如果需要）
        reg_loss = 0.0
        if eps_gen > 0:
            # 实现BooVAE的生成器正则化
            reg_loss = self.generator_regularization()
            loss = loss + eps_gen * reg_loss
        
        return loss, rec_loss, kl_loss, reg_loss
    
    def generator_regularization(self):
        """
        计算生成器正则化项
        
        返回:
            reg_loss: 正则化损失
        """
        # 实现BooVAE的生成器正则化
        # 对先前任务的伪输入进行正则化，防止当前任务的更新导致遗忘
        reg_loss = 0.0
        
        # 跳过第一个任务，因为没有先前任务
        if self.current_task_id == 0:
            return reg_loss
        
        # 对每个先前任务
        for task_id in range(self.current_task_id):
            # 获取任务的伪输入
            pseudo_inputs = self.prior.pseudo_inputs[task_id]
            
            # 对每个伪输入
            for i in range(pseudo_inputs.size(0)):
                # 确保伪输入形状正确
                pseudo_input = pseudo_inputs[i].unsqueeze(0)  # [1, n_input]
                
                # 使用当前编码器编码伪输入
                mu_curr, logsigma_curr = self.encoder(pseudo_input)
                
                # 正则化项：使当前编码结果接近之前的编码结果
                # 这需要先验中存储之前的编码结果
                # 此处简化为非对称KL散度
                # 完整实现需要存储之前的编码结果
                
                # 临时措施：使用先验的log_density作为正则化目标
                z_sample = self.prior.get_log_density(mu_curr, task_id)
                reg_loss += -z_sample.mean()
        
        return reg_loss
    
    def get_all_embeddings(self, adata, batch_size=128, emb_names=None, batch_col='batch', inplace=True):
        """
        获取所有嵌入
        
        参数:
            adata: AnnData对象
            batch_size: 批处理大小
            emb_names: 要返回的嵌入名称列表
            batch_col: 批次列名
            inplace: 是否直接修改adata
            
        返回:
            嵌入字典或None
        """
        from torch.utils.data import DataLoader, TensorDataset
        import numpy as np
        
        if emb_names is None:
            emb_names = ['delta', 'theta']
        
        # 准备数据
        if hasattr(adata.X, 'toarray'):
            X = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        else:
            X = torch.tensor(adata.X, dtype=torch.float32)
        
        # 获取批次信息
        if batch_col in adata.obs and self.encoder.use_batch_id:
            batch_indices = torch.tensor(adata.obs[batch_col].cat.codes.values, dtype=torch.long)
        else:
            batch_indices = None
        
        # 创建数据加载器
        if batch_indices is not None:
            dataset = TensorDataset(X, batch_indices)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        else:
            dataset = TensorDataset(X)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # 收集嵌入
        embeddings = {name: [] for name in emb_names}
        
        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                if batch_indices is not None:
                    x_batch, batch_idx = batch
                    batch_idx = batch_idx.to(self.device)
                else:
                    x_batch = batch[0]
                    batch_idx = None
                
                x_batch = x_batch.to(self.device)
                
                # 编码
                mu, logsigma, delta, theta = self.encode(x_batch, batch_idx)
                
                # 收集嵌入
                if 'delta' in emb_names:
                    embeddings['delta'].append(delta.cpu().numpy())
                if 'theta' in emb_names:
                    embeddings['theta'].append(theta.cpu().numpy())
                if 'mu' in emb_names:
                    embeddings['mu'].append(mu.cpu().numpy())
                if 'logsigma' in emb_names:
                    embeddings['logsigma'].append(logsigma.cpu().numpy())
        
        # 合并嵌入
        for name in embeddings:
            embeddings[name] = np.concatenate(embeddings[name], axis=0)
        
        # 获取主题-基因和基因-嵌入矩阵
        if inplace:
            # 将嵌入存储到adata中
            for name, emb in embeddings.items():
                adata.obsm[name] = emb
            
            # 存储主题-基因矩阵
            if hasattr(self.decoder, 'get_gene_embedding'):
                adata.varm['rho'] = self.decoder.get_gene_embedding().detach().cpu().numpy().T
            
            # 存储主题嵌入矩阵
            adata.uns['alpha'] = self.decoder.alpha.detach().cpu().numpy()
            
            return None
        else:
            return embeddings
    
    def save(self, save_dir):
        """
        保存模型
        
        参数:
            save_dir: 保存目录
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存模型状态
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'prior_state_dict': self.prior.state_dict(),
            'current_task_id': self.current_task_id,
            'model_params': {
                'norm_cells': self.norm_cells,
                'normed_loss': self.normed_loss,
                'normalize_beta': self.normalize_beta
            }
        }, os.path.join(save_dir, 'model.pt'))
        
        _logger.info(f'模型已保存到 {save_dir}')
    
    @classmethod
    def load(cls, save_dir, encoder_class, decoder_class, prior_class, device=None):
        """
        加载模型
        
        参数:
            save_dir: 保存目录
            encoder_class: 编码器类
            decoder_class: 解码器类
            prior_class: 先验类
            device: 计算设备
            
        返回:
            加载的模型
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型状态
        checkpoint = torch.load(os.path.join(save_dir, 'model.pt'), map_location=device)
        
        # 创建模型组件
        # 注意：这需要额外的模型配置信息，此处简化处理
        # 实际应用中应保存和加载完整的模型配置
        
        # 创建编码器
        encoder = encoder_class(...)  # 需要编码器参数
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        
        # 创建解码器
        decoder = decoder_class(...)  # 需要解码器参数
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        # 创建先验
        prior = prior_class(...)  # 需要先验参数
        prior.load_state_dict(checkpoint['prior_state_dict'])
        
        # 创建模型
        model = cls(
            encoder=encoder,
            decoder=decoder,
            prior=prior,
            **checkpoint['model_params'],
            device=device
        )
        
        # 设置当前任务ID
        model.current_task_id = checkpoint['current_task_id']
        
        _logger.info(f'从 {save_dir} 加载模型')
        
        return model
    
    def expand_prior_for_new_task(self, new_data, optimizer, num_steps=200, lambda_val=1.0):
        """
        为新任务扩展先验
        
        参数:
            new_data: 新任务数据
            optimizer: 优化器
            num_steps: 训练步骤数
            lambda_val: 权重参数
            
        返回:
            训练历史
        """
        # 增加当前任务ID
        self.current_task_id += 1
        
        # 扩展先验
        history = self.prior.expand_prior(new_data, optimizer, num_steps, lambda_val)
        
        return history
    
    def get_topic_word_matrix(self):
        """
        获取主题-词矩阵
        
        返回:
            主题-词矩阵
        """
        # 获取主题-基因矩阵
        gene_emb = self.decoder.get_gene_embedding()
        beta = torch.mm(self.decoder.alpha, gene_emb)
        
        if self.normalize_beta:
            beta = F.softmax(beta, dim=1)
            
        return beta.detach().cpu().numpy()
    
    def get_topics(self, gene_names, n_top_genes=10):
        """
        获取主题的顶部基因
        
        参数:
            gene_names: 基因名列表
            n_top_genes: 每个主题的顶部基因数量
            
        返回:
            主题-基因字典
        """
        beta = self.get_topic_word_matrix()
        topics = {}
        
        for topic_idx in range(beta.shape[0]):
            # 获取顶部基因索引
            top_gene_indices = np.argsort(-beta[topic_idx])[:n_top_genes]
            
            # 获取顶部基因名称
            top_genes = [gene_names[idx] for idx in top_gene_indices]
            
            # 存储主题基因
            topics[topic_idx] = top_genes
            
        return topics
    
    def get_cell_embeddings(self, adata, batch_col='batch', emb_name='theta'):
        """
        获取细胞嵌入
        
        参数:
            adata: AnnData对象
            batch_col: 批次列名
            emb_name: 嵌入名称
            
        返回:
            细胞嵌入
        """
        self.get_all_embeddings(adata, batch_col=batch_col, emb_names=[emb_name], inplace=True)
        return adata.obsm[emb_name]
    
    def get_reconstruction_error(self, adata, batch_col='batch', batch_size=128):
        """
        计算重构误差
        
        参数:
            adata: AnnData对象
            batch_col: 批次列名
            batch_size: 批处理大小
            
        返回:
            重构误差
        """
        from torch.utils.data import DataLoader, TensorDataset
        
        # 准备数据
        if hasattr(adata.X, 'toarray'):
            X = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        else:
            X = torch.tensor(adata.X, dtype=torch.float32)
        
        # 获取批次信息
        if batch_col in adata.obs and self.encoder.use_batch_id:
            batch_indices = torch.tensor(adata.obs[batch_col].cat.codes.values, dtype=torch.long)
        else:
            batch_indices = None
        
        # 创建数据加载器
        if batch_indices is not None:
            dataset = TensorDataset(X, batch_indices)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        else:
            dataset = TensorDataset(X)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # 计算重构误差
        total_error = 0.0
        total_samples = 0
        
        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                if batch_indices is not None:
                    x_batch, batch_idx = batch
                    batch_idx = batch_idx.to(self.device)
                else:
                    x_batch = batch[0]
                    batch_idx = None
                
                x_batch = x_batch.to(self.device)
                
                # 计算重构
                recon_x, _, _, _, _, _ = self.forward(x_batch, batch_idx)
                
                # 计算误差
                if self.normed_loss:
                    # 使用标准化数据计算误差
                    library_size = torch.sum(x_batch, dim=1, keepdim=True)
                    normed_x = x_batch / (library_size + 1e-7)
                    error = torch.sum((normed_x - recon_x).pow(2), dim=1).sum().item()
                else:
                    # 使用原始数据计算误差
                    error = torch.sum((x_batch - recon_x).pow(2), dim=1).sum().item()
                
                total_error += error
                total_samples += x_batch.size(0)
        
        return total_error / total_samples