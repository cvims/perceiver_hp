#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Organization  : Technische Hochschule Ingolstadt
# Institution   : Computer Vision for Intelligent Mobility Systems (CVIMS)
# Created By    : Dominik Rößle
# Created Date  : Friday April 08 hh:mm:ss GMT 2022
# Latest Update : Friday April 08 hh:mm:ss GMT 2022
# =============================================================================
"""
File description.
"""
# =============================================================================
# Imports
# =============================================================================
from torch import einsum
from torch.nn import Module, Linear, Dropout
from einops import rearrange
from .utils import default


__all__ = [
    'Attention'
]


class Attention(Module):
    """
    Adapted from: https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
    URL date: 15/10/2021
    """
    def __init__(self, query_dim: int, input_dim: int, dim_head: int, heads: int = 1, dropout: float = 0.0):
        super(Attention, self).__init__()
        att_dim = heads * dim_head
        input_dim = default(input_dim, query_dim)

        self.heads = heads

        # 1 / sqrt(dim_head)
        self.scale = dim_head ** -0.5

        # matrices to attention dimension
        self.query_transform = Linear(in_features=query_dim,
                                      out_features=att_dim,
                                      bias=False)

        # out_features * 2 because both key and value have the same dimensions (easier to calculate)
        self.key_value_transform = Linear(in_features=input_dim,
                                          out_features=att_dim * 2,
                                          bias=False)

        self.attn_dropout = Dropout(p=dropout)

        # cross attention output transform
        self.out_transform = Linear(
            in_features=att_dim,
            out_features=query_dim
        )

    def forward(self, x, context=None):
        context = default(context, x)

        query = self.query_transform(x)

        # split key and value in same size
        key, value = self.key_value_transform(context).chunk(2, dim=-1)

        query, key, value = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (query, key, value))

        head_scaling = einsum('b i d, b j d -> b i j', query, key) * self.scale

        attn = head_scaling.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # get back original shapes
        out = einsum('b i j, b j d -> b i d', attn, value)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)

        out = self.out_transform(out)

        return out
