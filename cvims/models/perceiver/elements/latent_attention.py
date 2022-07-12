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
from torch.nn import Module, LayerNorm
from .feed_forward import FeedForward
from .attention import Attention


__all__ = [
    'LatentAttention'
]


class LatentAttention(Module):
    def __init__(self, query_dim: int, dim_head: int, heads: int, dropout: float = 0.0,
                 feed_forward_dropout: float = 0.0):
        super(LatentAttention, self).__init__()

        # layer norm
        self.query_layer_norm = LayerNorm(query_dim)

        # latent attention
        self.latent_attention = Attention(
            query_dim=query_dim, input_dim=query_dim,
            heads=heads, dim_head=dim_head,
            dropout=dropout
        )

        # feed forward layer
        self.feed_forward_layer_norm = LayerNorm(query_dim)
        self.feed_forward = FeedForward(dim=query_dim, dropout=feed_forward_dropout)

    def forward(self, latents):
        latents_norm = self.query_layer_norm(latents)

        x = self.latent_attention(x=latents_norm) + latents

        ff_norm = self.feed_forward_layer_norm(x)
        x = self.feed_forward(ff_norm) + x

        return x
