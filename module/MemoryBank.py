"""
记忆库系统
用于存储真实样本(Real)的特征原型。
在推理(Single Mode)时，检索最相似的真实特征作为'基准'。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryBank(nn.Module):
    def __init__(self,feature_dim,
                 bank_size=1024,
                 momentum=0.9) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.bank_size = bank_size
        self.momentum = momentum

        # 注册缓冲区 -> 不t梯度更新但会随着模型进行保存
        # 初始化为随机正态分布并归一化

        self.register_buffer('features',torch.randn(bank_size,feature_dim))
        self.register_buffer('ptr',torch.zeros(1,dtype=torch.long))
        self.feature = F.normalize(self.features) # type: ignore
        
    def update(self,keys):
        