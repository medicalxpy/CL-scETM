import numpy as np
import scanpy as sc
import anndata
import torch
from scipy.sparse import issparse
import logging

_logger = logging.getLogger(__name__)

def preprocess_data(adata, 
                    n_top_genes=2000,
                    min_genes=200,
                    min_cells=3,
                    normalize_total=True,
                    log_transform=True,
                    target_sum=1e4):
    """
    预处理单细胞RNA-seq数据
    
    参数:
        adata: AnnData对象，包含单细胞RNA-seq数据
        n_top_genes: 选择的高变基因数量
        min_genes: 每个细胞至少表达的基因数
        min_cells: 每个基因至少在多少个细胞中表达
        normalize_total: 是否对每个细胞进行归一化
        log_transform: 是否进行对数转换
        target_sum: 归一化的目标和
    
    返回:
        处理后的AnnData对象
    """
    _logger.info('预处理数据...')
    
    # 过滤细胞和基因
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # 归一化和对数转换
    if normalize_total:
        sc.pp.normalize_total(adata, target_sum=target_sum)
    if log_transform:
        sc.pp.log1p(adata)
    
    # 找到高变基因
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable]
    
    return adata

def get_normalized_expression(adata):
    """
    获取归一化的表达矩阵
    
    参数:
        adata: 预处理后的AnnData对象
    
    返回:
        归一化表达矩阵的torch.Tensor
    """
    if issparse(adata.X):
        expr_matrix = adata.X.toarray()
    else:
        expr_matrix = adata.X
    
    return torch.FloatTensor(expr_matrix)

def split_train_test(adata, test_ratio=0.1, seed=42):
    """
    将数据分为训练集和测试集
    
    参数:
        adata: AnnData对象
        test_ratio: 测试集比例
        seed: 随机种子
    
    返回:
        train_adata, test_adata: 训练和测试AnnData对象
    """
    np.random.seed(seed)
    n_cells = adata.n_obs
    indices = np.arange(n_cells)
    np.random.shuffle(indices)
    test_size = int(n_cells * test_ratio)
    
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    
    train_adata = adata[train_idx].copy()
    test_adata = adata[test_idx].copy()
    
    return train_adata, test_adata

def get_cell_batches(adata, batch_col='batch'):
    """
    获取细胞批次信息
    
    参数:
        adata: AnnData对象
        batch_col: 批次列名
    
    返回:
        batch_indices: 批次索引的torch.Tensor
    """
    if batch_col not in adata.obs:
        _logger.warning(f'找不到批次列 {batch_col}，将创建单一批次')
        batch_indices = torch.zeros(adata.n_obs, dtype=torch.long)
    else:
        # 将批次标签转换为数值索引
        batch_labels = adata.obs[batch_col].astype('category')
        batch_indices = torch.tensor(batch_labels.cat.codes.values, dtype=torch.long)
    
    return batch_indices