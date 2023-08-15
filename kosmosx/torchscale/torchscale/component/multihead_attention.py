# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm

from .multiway_network import MultiwayWrapper
from .xpos_relative_position import XPOS

from inspect import isfunction
from einops import rearrange

from dataclasses import dataclass

from flash_attn import FlashAttention

#helpers
def exists(val):
    return val is not None


def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

@dataclass
class Intermediates:
    qk_similarities: Tensor = None
    pre_soft_max_attn: Tensor = None
    post_softmax_attn: Tensor = None

    def to_tuple(self):
        return (self.qk_similarities, self.pre_soft_max_attn, self.post_softmax_attn)
    

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

#https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, total_heads, **kwargs):
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads

        slopes = Tensor(self.__get_slopes(heads))
        slopes.register_buffer('slopes', slopes, persistent=False)
        self.register_bufferr('bias', None, persistent=False)

    def get_bias(self, i, j, device):
        i_arange = torch.arange(j -1, j, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias
    
    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log(heads).is_integer():
            return get_slopes_power_of_2(heads)
        
        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]
    
    @property
    def device(self):
        return next(self.buffers()).device
    
    def forward(self, i, j):
        h, device = self.self.total_heads, self.device
        
        if exists(self.bias) and self.bias.shape[-1] >= i and self.bias.shape[-2] >= i:
            return self.bias[..., :i, :j]
        
        bias = self.get_bias(i, j, device)
        bias = bias * self._get_slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim=0)
        self.register_buffer('bias', bias, persistent=False)

        return self.bias
    
class LearnedAlibiPositionalBias(AlibiPositionalBias):
    def __init__(self, heads, total_heads):
        super().__init__(heads, total_heads)
        log_slopes = torch.log(self.slopes)
        self.learned_logslopes = nn.Parameter(log_slopes)

    def forward(self, i, j):
        h, device = self.heads, self.device

        def get_slopes(param):
            return pad_at_dim(param.exp(), (0, h - param.shape[0]), dim=-2)
        
        if exists(self.bias) and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            bias = self.bias[..., :i, :j]
        else:
            bias = self.get_bias(i, j, device)
            self.register_buffer('bias', bias, persistent=False)
        
        slopes = get_slopes(self.learned_logslopes)
        bias = bias * slopes

        return bias
    



class MultiheadAttention(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        num_heads,
        dropout=0.0,
        self_attention=False,
        encoder_decoder_attention=False,
        subln=False,
        flash_attn=False,
        q_bucket_size = 512,
        k_bucket_size = 1024,
        alibi_pos_bias = False,
        alibi_num_heads = None,
        alibi_learned = False
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.flash_attn = flash_attn

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert self.self_attention ^ self.encoder_decoder_attention

        self.k_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.v_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.q_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.out_proj = MultiwayWrapper(
            args, nn.Linear(embed_dim, embed_dim, bias=True)
        )
        self.inner_attn_ln = (
            MultiwayWrapper(args, LayerNorm(self.embed_dim, eps=args.layernorm_eps))
            if subln and self.self_attention
            else None
        )

        self.dropout_module = torch.nn.Dropout(dropout)
        self.xpos = (
            XPOS(self.head_dim, args.xpos_scale_base)
            if args.xpos_rel_pos and self.self_attention
            else None
        )

        if flash_attn:
            self.flash_attention = FlashAttention(dim = embed_dim, heads = num_heads, q_bucket_size = q_bucket_size, k_bucket_size = k_bucket_size)

        if alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, self.num_heads)
            assert alibi_num_heads <= self.num_heads, 'number of ALiBi heads must be less than the total number of heads'
            alibi_pos_klass = LearnedAlibiPositionalBias if self.alibi_learned else AlibiPositionalBias
            self.rel_pos = alibi_pos_klass(heads = alibi_num_heads, total_heads=self.num_heads)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.flash_attn:
            q = rearrange(q, 'b h n d -> (b h) n d')
            k = rearrange(k, 'b h n d -> (b h) n d')
            v = rearrange(v, 'b h n d -> (b h) n d')

            if key_padding_mask is not None:
                key_padding_mask = rearrange(key_padding_mask, 'b n -> b () n ()').expand(-1, self.num_heads, -1, -1)
                key_padding_mask = rearrange(key_padding_mask, '(b h) n m -> (b h) n m')

            if attn_mask is not None:
                attn_mask = rearrange(attn_mask, 'b n m -> b () n m').expand(-1, self.num_heads, -1, -1)
                attn_mask = rearrange(attn_mask, '(b h) n m -> (b h) n m')

            out = self.flash_attention(q, k, v, mask = key_padding_mask)

        else:
            q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
            k = k.reshape(bsz * self.num_heads, src_len, self.head_dim)
            v = v.reshape(bsz * self.num_heads, src_len, self.head_dim)

            attn_weights = torch.bmm(q, k.transpose(1, 2))

            if attn_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    attn_mask.unsqueeze(0),
                    float("-inf"),
                )
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
            if exists(self.rel_pos):
                rel_pos = self.rel_pos.view(attn_weights.size())
                attn_weights = attn_weights + rel_pos

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
            attn_probs = self.dropout_module(attn_weights)
            out = torch.bmm(attn_probs, v)

        out = rearrange(out, '(b h) n d -> b h n d', h = self.num_heads)

        if self.inner_attn_ln is not None:
            out = self.inner_attn_ln(out)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)
        return out
