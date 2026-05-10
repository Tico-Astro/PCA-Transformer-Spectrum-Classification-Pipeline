import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, seq_len, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        max_len = max(5000, seq_len)
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)

        

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos((position * div_term))
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0: -1]

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    
    def forward(self, x):

        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)



'''
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        max_len = max(5000, seq_len)
        self.dropout = nn.Dropout(p=dropout)

        # 初始化位置编码矩阵
        self.max_len = max_len
        self.d_model = d_model

        # 预计算分母的常数部分
        self.div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 在注册之前检查是否已有同名属性
        if not hasattr(self, 'div_term'):
            self.register_buffer('div_term', self.div_term)

    def forward(self, x):
        """
        输入 x 的形状为 (batch_size, seq_len, d_model)
        动态生成相位并将其引入到正余弦函数中。
        """
        batch_size, seq_len, d_model = x.size()
       
        # 生成位置序列
        position = torch.arange(0, seq_len, dtype=torch.float,device='cuda:0').unsqueeze(1)  # (seq_len, 1)
       
        position.to('cuda:0').device

        # 使用输入数据动态生成相位项
        phase_shift = (x[:, 0, :].to('cuda:0') - x[:, 5, :].to('cuda:0'))
        # 取第一个时间步 (batch_size, d_model)
        phase_shift = phase_shift.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, d_model)

        
        

        
        #print(phase_shift.device)
        

        # 动态位置编码计算
        pe = torch.zeros(batch_size, seq_len, d_model,device='cuda:0')
        div_term = self.div_term.to("cuda:0") # 扩展到 batch_size
        
        pe[:, :, 0::2] = torch.sin(position * div_term) * torch.exp(phase_shift[:, :, 0::2])  # 加入相位到 sin
        if d_model % 2 == 0:
            pe[:, :, 1::2] = torch.cos(position * div_term)* torch.exp(phase_shift[:, :, 1::2])  # 加入相位到 cos
        else:
            pe[:, :, 1::2] = torch.cos(position * div_term )* torch.exp(phase_shift[:, :, 1::2])[:, :, :-1]

        # 对输入叠加位置编码
        x = x 
        return self.dropout(x)
'''