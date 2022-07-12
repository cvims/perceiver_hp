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
    'CrossAttention'
]


class CrossAttention(Module):
    def __init__(self, latent_dim: int, input_dim: int, cross_dim_head: int, cross_heads: int = 1,
                 cross_dropout: float = 0.0, feed_forward_dropout: float = 0.0):
        super(CrossAttention, self).__init__()

        # layer norm
        self.latent_layer_norm = LayerNorm(latent_dim)
        self.input_layer_norm = LayerNorm(input_dim)

        # cross attention
        self.cross_attention = Attention(
            query_dim=latent_dim, input_dim=input_dim,
            heads=cross_heads, dim_head=cross_dim_head,
            dropout=cross_dropout
        )

        # feed forward layer
        self.feed_forward_layer_norm = LayerNorm(latent_dim)
        self.feed_forward = FeedForward(
            dim=latent_dim, dropout=feed_forward_dropout
        )

    def forward(self, latents, data):
        latents_norm = self.latent_layer_norm(latents)
        data_norm = self.input_layer_norm(data)

        x = self.cross_attention(x=latents_norm, context=data_norm) + latents

        ff_norm = self.feed_forward_layer_norm(x)
        x = self.feed_forward(ff_norm) + x

        return x
