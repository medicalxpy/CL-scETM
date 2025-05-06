import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """
    线性解码器，将主题混合解码为基因表达
    基于scETM的解码器，保持矩阵三因子分解架构
    """
    
    def __init__(self, 
                 n_topics, 
                 n_genes, 
                 trainable_gene_emb_dim=400,
                 fixed_gene_emb=None,
                 enable_batch_bias=True,
                 enable_global_bias=False,
                 n_batches=1):
        """
        初始化解码器
        
        参数:
            n_topics: 主题数量
            n_genes: 基因数量
            trainable_gene_emb_dim: 可训练基因嵌入维度
            fixed_gene_emb: 固定基因嵌入，如通路信息 [fixed_dim, n_genes]
            enable_batch_bias: 是否启用批次偏置
            enable_global_bias: 是否启用全局偏置
            n_batches: 批次数量
        """
        super(Decoder, self).__init__()
        
        self.n_topics = n_topics
        self.n_genes = n_genes
        self.trainable_gene_emb_dim = trainable_gene_emb_dim
        self.enable_batch_bias = enable_batch_bias
        self.enable_global_bias = enable_global_bias
        self.n_batches = n_batches
        
        # 主题嵌入矩阵 alpha [n_topics, emb_dim]
        self.fixed_gene_emb_dim = 0
        if fixed_gene_emb is not None:
            self.fixed_gene_emb_dim = fixed_gene_emb.shape[0]
            self.register_buffer('fixed_gene_emb', fixed_gene_emb)
        
        # 总嵌入维度
        total_emb_dim = self.trainable_gene_emb_dim + self.fixed_gene_emb_dim
        
        # 初始化主题嵌入矩阵
        self.alpha = nn.Parameter(torch.randn(n_topics, total_emb_dim))
        
        # 初始化可训练基因嵌入矩阵
        if trainable_gene_emb_dim > 0:
            self.trainable_gene_emb = nn.Parameter(torch.randn(trainable_gene_emb_dim, n_genes))
        else:
            self.register_parameter('trainable_gene_emb', None)
        
        # 批次偏置 [n_batches, n_genes]
        if enable_batch_bias:
            self.batch_bias = nn.Parameter(torch.zeros(n_batches, n_genes))
        else:
            self.register_parameter('batch_bias', None)
        
        # 全局偏置 [1, n_genes]
        if enable_global_bias:
            self.global_bias = nn.Parameter(torch.zeros(1, n_genes))
        else:
            self.register_parameter('global_bias', None)
            
    def get_gene_embedding(self):
        """获取完整的基因嵌入矩阵"""
        gene_emb_parts = []
        
        if self.trainable_gene_emb is not None:
            gene_emb_parts.append(self.trainable_gene_emb)
            
        if hasattr(self, 'fixed_gene_emb'):
            gene_emb_parts.append(self.fixed_gene_emb)
            
        return torch.cat(gene_emb_parts, dim=0) if gene_emb_parts else None
    
    def forward(self, theta, batch_indices=None, normalize_beta=False):
        """
        前向传播
        
        参数:
            theta: 主题混合 [batch_size, n_topics]
            batch_indices: 批次索引 [batch_size]
            normalize_beta: 是否标准化beta矩阵
            
        返回:
            recon_x: 重构的基因表达 [batch_size, n_genes]
        """
        # 获取基因嵌入
        gene_emb = self.get_gene_embedding()
        
        # 计算beta矩阵 = alpha @ rho
        beta = torch.mm(self.alpha, gene_emb)  # [n_topics, n_genes]
        
        if normalize_beta:
            # 如果需要标准化beta
            beta = F.softmax(beta, dim=1)
            recon_x = torch.mm(theta, beta)
        else:
            # 计算重构: theta @ beta
            recon_logit = torch.mm(theta, beta)  # [batch_size, n_genes]
            
            # 添加偏置项
            if self.enable_global_bias and self.global_bias is not None:
                recon_logit = recon_logit + self.global_bias
                
            if self.enable_batch_bias and self.batch_bias is not None and batch_indices is not None:
                recon_logit = recon_logit + self.batch_bias[batch_indices]
                
            # 应用softmax获得基因表达概率
            recon_x = F.softmax(recon_logit, dim=1)
        
        return recon_x