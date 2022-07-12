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
Implementation for the Perceiver architecture
"""
# =============================================================================
# Imports
# =============================================================================
import torch
from torch.nn import Module, Parameter, ModuleList, init
from einops import rearrange, repeat

from .elements import CrossAttention, perceiver_output_to_logits
from .elements.latent_attention import LatentAttention
from .elements.utils import fourier_encode


__all__ = [
    'Perceiver', 'MultiInputPerceiver'
]


class Perceiver(Module):
    """
    Architecture from the Perceiver Deep Mind paper.
    Adapted from: https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
    URL date: 15/10/2021
    """

    def __init__(
            self,
            iterative_count: int,
            input_channels: int,
            input_axis: int,
            num_latents: int,
            latent_dim: int,
            self_per_cross_attention: int,
            cross_dim_head: int,
            latent_dim_head: int,
            cross_heads: int = 1,
            latent_heads: int = 1,
            tie_weight_pos_cross_attention: int = -1,
            tie_weight_pos_latent_attention: int = -1,
            cross_attention_dropout: float = 0.,
            latent_attention_dropout: float = 0.,
            cross_attention_feed_forward_dropout: float = .0,
            latent_attention_feed_forward_dropout: float = .0,
            fourier_encode_data: bool = False,
            num_freq_bands: int = 6,
            max_freq: float = 10.,
            output_classes: int = None
    ):
        """

        :param iterative_count: Iterative attention count
        :param input_channels: Input channels of considered modality
        :param input_axis: Axis (index dimensions), e.g. for images 2 for width x height
        :param num_latents: Latent representation size
        :param latent_dim: Latent dimension size
        :param self_per_cross_attention: Self-attention chain size after cross attention module
        :param cross_dim_head: Attention dimension size per head for cross-attention
        :param latent_dim_head: Attention dimension size per head for self-attention
        :param cross_heads: Attention heads of the cross-attention module
        :param latent_heads: Attention heads of the self(latent)-attention module
        :param tie_weight_pos_cross_attention: Weight sharing for cross attention from specific pos until end of
        network. Use -1 to not share weights at all. Weight sharing starts from parameter value smaller or equal
        to 2 since the first 'shared' weights can only be applied after the initialization of the first cross-attention
        module.
        :param tie_weight_pos_latent_attention: Weight sharing for self attention from specific pos until end of
        network. Use -1 to not share weights at all. Weight sharing starts from parameter value smaller or equal
        to 2 since the first 'shared' weights can only be applied after the initialization of the first cross-attention
        module.
        :param cross_attention_dropout: Dropout rate for the linear layers for cross-attention
        :param latent_attention_dropout: Dropout rate for the linear layers for self-attention
        :param cross_attention_feed_forward_dropout: Dropout rate for the linear output layers of cross-attention
        :param latent_attention_feed_forward_dropout: Dropout rate for the linear output layers of self-attention
        :param fourier_encode_data: Boolean True | False if fourier encodings are to be generated and applied
        :param num_freq_bands: Number of bands for fourier encoding
        :param max_freq: Maximal frequency for fourier encoding
        :param output_classes: Number of output classes for classification. If None then the perceiver output will be
        provided raw
        """
        super(Perceiver, self).__init__()

        self.num_latents = num_latents
        self.input_axis = input_axis
        self.input_channels = input_channels

        self.fourier_encode_data = fourier_encode_data
        self.num_freq_bands = num_freq_bands
        self.max_freq = max_freq

        # init normal latents
        latents = torch.normal(mean=0.0, std=0.02, size=(num_latents, latent_dim))
        latents = torch.clip_(latents, min=-2, max=2)
        self.latents = Parameter(data=latents, requires_grad=True)

        fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        input_dim = fourier_channels + self.input_channels

        # cross-attention layer
        def cross_attention_block():
            return CrossAttention(
                latent_dim=latent_dim, input_dim=input_dim,
                cross_heads=cross_heads, cross_dim_head=cross_dim_head,
                cross_dropout=cross_attention_dropout,
                feed_forward_dropout=cross_attention_feed_forward_dropout
            )

        # self(latent)-attention layer
        def self_attention_layer():
            return LatentAttention(
                query_dim=latent_dim,
                heads=latent_heads, dim_head=latent_dim_head,
                dropout=latent_attention_dropout,
                feed_forward_dropout=latent_attention_feed_forward_dropout
            )

        # create modules list to combine cross attention and self(latent)-attention
        # weight sharing
        self.layers = ModuleList([])

        for i in range(iterative_count):
            # share weights if param is set
            # share cross attention weights
            share_cross_weights = True if i >= tie_weight_pos_cross_attention - 1 and i != 0 and tie_weight_pos_cross_attention > -1 else False
            # share latent attention weights
            share_latent_weights = True if i >= tie_weight_pos_latent_attention - 1 and i != 0 and tie_weight_pos_latent_attention > -1 else False

            # use the previous layers if share weights is True
            subsequent_blocks = ModuleList([])
            if self.layers.__len__() > 0 and share_cross_weights:
                subsequent_blocks.append(self.layers[-1][0])
            else:
                subsequent_blocks.append(cross_attention_block())

            if self.layers.__len__() > 0 and share_latent_weights:
                subsequent_blocks.append(self.layers[-1][1])
            else:
                latent_blocks = ModuleList([])
                # self(latent)- attention
                for _ in range(self_per_cross_attention):
                    latent_blocks.append(self_attention_layer())

                subsequent_blocks.append(latent_blocks)

            # add them as one 'block' to the layers module list
            self.layers.append(subsequent_blocks)

        self.output = None
        if output_classes:
            self.output = perceiver_output_to_logits(
                dim=latent_dim, num_classes=output_classes
            )
        else:
            self.output = torch.nn.Identity()

    def forward(self, data):
        # indices => bs: 0, channels: 1, *axis: 2+
        batch_size, channels, *axis = data.shape
        device = data.device
        d_type = data.dtype
        data = rearrange(data, 'b d ... -> b ... d')

        assert len(axis) == self.input_axis, 'input data must have the right number of axis'
        assert channels == self.input_channels, 'data channels must have the right number of initial set input channels'

        # fourier encoding
        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis
            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=d_type), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=batch_size)

            data = torch.cat((data, enc_pos), dim=-1)

        # combine inner dimensions / make them ready for transformers
        # flatten the index dimensions
        data = rearrange(data, 'b ... d -> b (...) d')
        # repeat the latents for all batches
        x = repeat(self.latents, 'n d -> b n d', b=batch_size).to(device=device)

        for cross_attn, latent_blocks in self.layers:
            # cross attention
            x = cross_attn(latents=x, data=data)

            # latent attention
            for latent_attn in latent_blocks:
                x = latent_attn(latents=x)

        return self.output(x)


class MultiInputPerceiver(Perceiver):
    """
    Architecture for a single modality with dynamic amount of instances.
    Parameters and architecture is inherited from the Perceiver architecture.
    """
    def __init__(self, **kwargs):
        # we handle output classes separately
        multi_instance_classification_output_classes = None
        if 'output_classes' in kwargs:
            multi_instance_classification_output_classes = kwargs['output_classes']
            del kwargs['output_classes']
        super(MultiInputPerceiver, self).__init__(**kwargs)

        self.multi_instance_output = None
        if multi_instance_classification_output_classes:
            self.multi_instance_output = perceiver_output_to_logits(
                dim=kwargs['latent_dim'], num_classes=multi_instance_classification_output_classes,
                is_multi_instance=True
            )
        else:
            self.multi_instance_output = torch.nn.Identity()

    def forward(self, data):
        # Rearrange the instance dimension first and apply the usual Perceiver forward pass
        # 0: batch size, 1: instance size, 2: channels, 3+: *axis
        instances = data.shape[1]

        # combine batch and instance dimension
        data = rearrange(data, 'b i ... -> (b i) ...')

        # apply the Perceiver forward pass
        x = super().forward(data=data)

        # Convert back the initial data shape
        x = rearrange(x, '(b i) n d -> b i n d', i=instances)

        if self.multi_instance_output:
            x = self.multi_instance_output(x)

        return x
