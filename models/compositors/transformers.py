import torch
import torch.nn as nn

from models.attention_modules.self_attention import AttentionModule
from models.compositors.ibn import IBN


class DisentangledTransformer(nn.Module):
    def __init__(self, feature_size, text_feature_size, num_heads, norm='in', global_styler=None, *args, **kwargs):
        super().__init__()
        self.n_heads = num_heads
        self.c_per_head = feature_size // num_heads
        assert feature_size == self.n_heads * self.c_per_head

        self.att_module = AttentionModule(feature_size, text_feature_size, num_heads, *args, **kwargs)
        self.att_module2 = AttentionModule(feature_size, text_feature_size, num_heads, *args, **kwargs)
        #self.att_module3 = AttentionModule(feature_size, text_feature_size, num_heads, *args, **kwargs)
        self.global_styler = global_styler

        self.weights = nn.Parameter(torch.tensor([1., 1.]))
        if norm == 'in':
            self.instance_norm = nn.InstanceNorm2d(feature_size)
        elif norm == 'bn':
            self.instance_norm = nn.BatchNorm2d(feature_size)
        elif norm == 'ibn':
            self.instance_norm = IBN(feature_size)

    def forward(self, x, t, *args, **kwargs):
        normed_x = self.instance_norm(x)
        att_out, att_map = self.att_module(normed_x, t, return_map=True)
        out = normed_x + self.weights[0] * att_out
        #out = self.weights[0] * att_out

        att_out2, att_map2 = self.att_module2(out, t, return_map=True)
        out = out + self.weights[1] * att_out2

        #att_out3, att_map3 = self.att_module3(out, t, return_map=True)
        #out = out + self.weights[2] * att_out3

        if self.global_styler != None:
            out = self.global_styler(out, t, x=x)

        #out = x + out
        return out, att_map
