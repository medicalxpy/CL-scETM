import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    将单细胞数据编码为主题混合的编码器网络
    基于scETM的编码器，但集成了持续学习能力
    """
    
    def __init__(self, 
                 n_input, 
                 n_topics, 
                 hidden_sizes=(128,), 
                 bn=True,
                 dropout_prob=0.1,
                 use_batch_id=False,
                 n_batches=1):
        """
        初始化编码器
        
        参数:
            n_input: 输入特征数（基因数）
            n_topics: 主题数量
            hidden_sizes: 隐藏层大小的元组
            bn: 是否使用批量归一化
            dropout_prob: Dropout概率
            use_batch_id: 是否使用批次ID作为输入
            n_batches: 批次数量
        """
        super(Encoder, self).__init__()
        
        self.n_input = n_input
        self.n_topics = n_topics
        self.hidden_sizes = hidden_sizes
        self.bn = bn
        self.dropout_prob = dropout_prob
        self.use_batch_id = use_batch_id
        self.n_batches = n_batches
        
        # 构建编码器网络
        layers = []
        
        # 第一层
        input_dim = n_input + (n_batches - 1 if use_batch_id else 0)
        layers.append(nn.Linear(input_dim, hidden_sizes[0]))
        layers.append(nn.ReLU())
        if bn:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        if dropout_prob > 0:
            layers.append(nn.Dropout(dropout_prob))
        
        # 添加额外的隐藏层
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            if bn:
                layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
        
        self.encoder_network = nn.Sequential(*layers)
        
        # 输出层 - 主题混合参数 (delta)
        self.mu_q_delta = nn.Linear(hidden_sizes[-1], n_topics)
        self.logsigma_q_delta = nn.Linear(hidden_sizes[-1], n_topics)
        
    def forward(self, x, batch_indices=None):
        """
        前向传播
        
        参数:
            x: 输入数据 [batch_size, n_input]
            batch_indices: 批次索引 [batch_size]
            
        返回:
            mu_q_delta: 主题混合均值
            logsigma_q_delta: 主题混合对数方差
        """
        # 如果需要批次ID，则添加到输入中
        if self.use_batch_id and batch_indices is not None:
            batch_onehot = F.one_hot(batch_indices, self.n_batches)
            # 删除第一个类别（参考类别）
            batch_onehot = batch_onehot[:, 1:].float()
            x = torch.cat([x, batch_onehot], dim=1)
        
        # 通过编码器网络
        q_delta = self.encoder_network(x)
        
        # 获取参数
        mu_q_delta = self.mu_q_delta(q_delta)
        logsigma_q_delta = self.logsigma_q_delta(q_delta)
        
        # 限制logsigma以防止数值不稳定
        logsigma_q_delta = torch.clamp(logsigma_q_delta, min=-10, max=10)
        
        return mu_q_delta, logsigma_q_delta