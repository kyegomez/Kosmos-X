# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm

from .multiway_network import MultiwayWrapper
from .xpos_relative_position import XPOS
from einops.layers.torch import Rearrange

from einops import rearrange, repeat, reduce

from dataclasses import dataclass

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
        # flash_attn=False,
        alibi_pos=False,
        # one_write_head=False,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

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

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)


    # def flash_attn(
    #     self,
    #     q, k, v,
    #     mask=None,
    #     attn_bias=None
    # ):
    #     batch, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

    #     #recommended for multi query single key value attention
    #     #kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

    #     if k.ndim == 3:
    #         k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)
    #     if v.ndim == 3:
    #         v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

    #     #handle scale by default they scale by dim_head ** -0.5 but need to take care if using cosine sim attention

    #     if self.qk_norm:
    #         default_scale = q.shape[-1] ** -0.5
    #         q = q * (default_scale / self.scale)

    #     #check if mask exists and expand to compatible shape
    #     #the mask is B L so it would have to be expanded to B H N L

    #     casual = casual

    #     if exists(mask):
    #         assert mask.ndim == 4
    #         mask = mask.expand(batch, heads, q_len, k_len)

    #         #manually handle casual mask, if another mask was given
            
    #         if casual:
    #             casual_mask = torch.ones((q_len, k_len), dtype = torch.bool, device=device).triu(k_len - q_len + 1)
    #             mask = mask | casual_mask
    #             casual = False
        
    #     #handle alibi positional bias
    #     #convet from bool to gloat
        
    #     if exists(attn_bias):
    #         attn_bias = rearrange(attn_bias, 'h i j -> 1 h i j').expand(batch, -1, -1, -1)

    #         #if mask given, the mask would already contain the casual_mask from above logic
    #         #otherwise if no mask given but still contain casual, mask out alibi positional bias to a large negative number

    #         mask_value = - torch.finfo(q.dtype).max

    #         if exists(mask):
    #             attn_bias = attn_bias.masked_fill(mask, mask_value // 2)
    #         elif casual:
    #             casual_mask = torch.ones((q_len, k_len), dtype = torch.bool, device = device).triu(k_len - q_len + 1)
    #             attn_bias = attn_bias.masked_fill(casual_mask, mask_value // 2)
    #             casual = False
    #         #scaled_dot_product_attention handles attn_mask either as bool or additive vias
    #         #make it an addiitve bias

    #         mask = attn_bias

        
    #     #checlk if there is a compatible device for flash attention

    #     config = self.cuda_config if is_cuda else self.cpu_config

    #     #pytorch 2.0a flash attention: q, k, v, mask, dropout, casual, softmax_scale

    #     with torch.backends.cuda.sdp_kernel(**config._asdict()):
    #         out = F.scaled_dot_product_attention(
    #             q, k, v,
    #             attn_mask = mask,
    #             dropout_p = self.dropout if self.training else 0.,
    #             is_casual = casual
    #         )

    #     return out, Intermediates()
    

    def forward(
        self,
        query,
        key,
        value,
        incremental_state=None,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
    ):
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz * self.num_heads, src_len, self.head_dim)
        v = v.reshape(bsz * self.num_heads, src_len, self.head_dim)

        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                prev_value = incremental_state["prev_value"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
            incremental_state["prev_key"] = k.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            incremental_state["prev_value"] = v.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            src_len = k.size(1)

        if self.xpos is not None:
            if incremental_state is not None:
                offset = src_len - 1
            else:
                offset = 0
            k = self.xpos(k, offset=0, downscale=True)
            q = self.xpos(q, offset=offset, downscale=False)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        if self.alibi_pos:
            alibi_bias = self.alibi_positional_bias(tgt_len, src_len)
            attn_weights = attn_weights + alibi_bias
            

        if rel_pos is not None:
            rel_pos = rel_pos.view(attn_weights.size())
            attn_weights = attn_weights + rel_pos

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).reshape(tgt_len, bsz, embed_dim).transpose(0, 1)

        if self.inner_attn_ln is not None:
            attn = self.inner_attn_ln(attn)

        attn = self.out_proj(attn)
        attn_weights = attn_weights.view(
            bsz, self.num_heads, tgt_len, src_len
        ).transpose(1, 0)

        return attn, attn_weights
