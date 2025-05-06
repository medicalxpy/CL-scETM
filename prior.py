import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ExpandablePrior(nn.Module):
    """
    可扩展先验分布，基于BooVAE的持续学习方法
    使用伪输入和编码器来表示先验分布
    """
    
    def __init__(self, encoder, n_components=10):
        """
        初始化可扩展先验
        
        参数:
            encoder: 编码器模型
            n_components: 每个任务的最大组件数
        """
        super(ExpandablePrior, self).__init__()
        
        self.encoder = encoder
        self.n_components = n_components
        
        # 伪输入列表，每个任务一个列表
        self.pseudo_inputs = nn.ParameterList()
        
        # 组件权重
        self.component_weights = nn.ParameterList()
        
        # 用于记住上一个任务的参数
        self.prev_encoder_params = None
        
        # 任务计数
        self.num_tasks = 0
        
        # 初始化第一个任务
        self._init_new_task()
        
    def _init_new_task(self):
        """初始化新任务的参数"""
        
        # 为新任务添加伪输入参数
        pseudo_inputs = nn.Parameter(
            torch.randn(self.n_components, self.encoder.n_input) * 0.1
        )
        self.pseudo_inputs.append(pseudo_inputs)
        
        # 初始化新任务的组件权重
        weights = nn.Parameter(
            torch.ones(self.n_components) / self.n_components
        )
        self.component_weights.append(weights)
        
        # 增加任务计数
        self.num_tasks += 1
        
    def add_component(self, task_id=None):
        """
        使用编码器添加新组件到先验
        
        参数:
            task_id: 要添加组件的任务ID，如果为None则为最新任务
        """
        if task_id is None:
            task_id = self.num_tasks - 1
            
        if task_id >= self.num_tasks or task_id < 0:
            raise ValueError(f"任务ID {task_id} 超出范围 [0, {self.num_tasks-1}]")
        
        # 训练新组件前先存储当前编码器参数
        self.store_encoder_params()
    
    def store_encoder_params(self):
        """存储当前编码器参数"""
        self.prev_encoder_params = {}
        for name, param in self.encoder.named_parameters():
            self.prev_encoder_params[name] = param.detach().clone()
    
    def restore_encoder_params(self):
        """恢复之前的编码器参数"""
        if self.prev_encoder_params is not None:
            for name, param in self.encoder.named_parameters():
                if name in self.prev_encoder_params:
                    param.data = self.prev_encoder_params[name]
    
    def get_current_task_pseudo_inputs(self):
        """获取当前任务的伪输入"""
        return self.pseudo_inputs[-1]
    
    def get_log_density(self, z, task_id=None):
        """
        计算先验对数密度 log p(z)
        
        参数:
            z: 潜变量 [batch_size, latent_dim]
            task_id: 任务ID，如果为None则考虑所有任务
            
        返回:
            log_density: 对数密度 [batch_size]
        """
        log_densities = []
        
        if task_id is None:
            # 考虑所有任务
            tasks_range = range(self.num_tasks)
        else:
            # 只考虑特定任务
            tasks_range = [task_id]
            
        # 计算每个任务的对数密度
        for t in tasks_range:
            task_log_density = self._get_task_log_density(z, t)
            log_densities.append(task_log_density)
            
        # 合并所有任务的对数密度（考虑任务权重）
        if len(log_densities) == 1:
            return log_densities[0]
        else:
            # 假设所有任务权重相等
            stacked_densities = torch.stack(log_densities, dim=1)
            return torch.logsumexp(stacked_densities, dim=1) - np.log(len(tasks_range))
    
    def _get_task_log_density(self, z, task_id):
        """
        计算特定任务的先验对数密度
        
        参数:
            z: 潜变量 [batch_size, latent_dim]
            task_id: 任务ID
            
        返回:
            log_density: 对数密度 [batch_size]
        """
        # 获取任务的伪输入和权重
        pseudo_inputs = self.pseudo_inputs[task_id]
        weights = F.softmax(self.component_weights[task_id], dim=0)
        
        log_densities = []
        
        # 使用编码器计算每个伪输入的密度
        for i in range(self.n_components):
            # 确保伪输入形状正确
            pseudo_input = pseudo_inputs[i].unsqueeze(0)  # [1, n_input]
            
            # 保存并恢复编码器参数以防止更新
            self.store_encoder_params()
            
            # 通过编码器获取分布参数
            with torch.no_grad():
                mu, logsigma = self.encoder(pseudo_input)
            
            # 恢复编码器参数
            self.restore_encoder_params()
            
            # 计算负平方马氏距离
            diff = z - mu
            log_density = -0.5 * torch.sum(
                torch.exp(-logsigma) * diff * diff + logsigma + np.log(2 * np.pi),
                dim=1
            )
            
            log_densities.append(log_density + torch.log(weights[i] + 1e-10))
            
        # 合并所有组件的对数密度
        stacked_densities = torch.stack(log_densities, dim=1)
        return torch.logsumexp(stacked_densities, dim=1)
    
    def expand_prior(self, new_data, optimizer, num_steps=200, lambda_val=1.0):
        """
        扩展先验分布以适应新任务
        
        参数:
            new_data: 新任务数据 [batch_size, n_input]
            optimizer: 优化器
            num_steps: 训练步骤数
            lambda_val: 权重参数
            
        返回:
            训练历史
        """
        self._init_new_task()
        
        # 训练历史
        history = {'loss': []}
        
        # 存储之前所有任务的先验分布
        prev_task_prior = None
        if self.num_tasks > 1:
            with torch.no_grad():
                z = torch.randn(1000, self.encoder.n_topics).to(next(self.parameters()).device)
                prev_task_prior = self.get_log_density(z, task_id=self.num_tasks-2)
        
        # 训练伪输入
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # 获取当前伪输入
            pseudo_inputs = self.get_current_task_pseudo_inputs()
            
            # 计算目标先验
            with torch.no_grad():
                # 从新数据采样
                indices = torch.randint(0, new_data.size(0), (min(100, new_data.size(0)),))
                data_sample = new_data[indices]
                
                # 编码样本
                mu, logsigma = self.encoder(data_sample)
                
                # 采样潜变量
                epsilon = torch.randn_like(mu)
                z = mu + epsilon * torch.exp(0.5 * logsigma)
            
            # 计算当前先验
            curr_log_density = self._get_task_log_density(z, self.num_tasks-1)
            
            # 计算损失
            loss = -torch.mean(curr_log_density)
            
            # 如果有之前的任务，添加正则化项
            if prev_task_prior is not None:
                # 计算与之前任务先验的KL散度
                with torch.no_grad():
                    prev_log_density = prev_task_prior
                kl_term = torch.mean(curr_log_density - prev_log_density)
                loss = loss + lambda_val * kl_term
            
            # 更新伪输入
            loss.backward()
            optimizer.step()
            
            # 记录历史
            history['loss'].append(loss.item())
            
        return history
    
    def forward(self, z, task_id=None):
        """
        计算先验对数密度 log p(z)
        
        参数:
            z: 潜变量 [batch_size, latent_dim]
            task_id: 任务ID，如果为None则考虑所有任务
            
        返回:
            log_density: 对数密度 [batch_size]
        """
        return self.get_log_density(z, task_id)