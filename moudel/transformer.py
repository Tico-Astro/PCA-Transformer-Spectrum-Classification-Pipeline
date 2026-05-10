import torch, copy
import torch.nn as nn

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoders(nn.Module):
    def __init__(self, encoder_layer, num_layers, device=None, norm=None):
        super(TransformerEncoders, self).__init__()
        # 1. 移除硬编码的 'cuda:0'，默认设为 None
        self.device = device
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        
        # 2. 【核心修改】动态获取设备
        # 如果 self.device 为 None，或者为了确保安全，直接使用输入数据 src 的设备
        curr_device = self.device if self.device is not None else src.device
        
        # 3. 使用动态设备创建零张量
        attn_output = torch.zeros(
            (src.shape[0], src.shape[1], src.shape[1]), 
            device=curr_device
        )
        
        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_output += attn  

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_output





























# pe)