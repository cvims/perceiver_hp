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
from torch.nn import Sequential, LayerNorm, Linear
from einops.layers.torch import Reduce


__all__ = [
    'perceiver_output_to_logits'
]


def perceiver_output_to_logits(dim, num_classes, is_multi_instance=False, reduction: str = 'mean'):
    """
    Uses the Perceiver outputs and projects it to logits.
    :param dim: last dimension size (shape[-1])
    :param num_classes: output neurons
    :param is_multi_instance: einops transforming True if len(shape)==4; False if len(shape)==3
    :param reduction: Possible values: 'min', 'max', 'sum', 'mean', 'prod'
    :return: Torch sequential block
    """
    ein_op = 'b i n d -> b d' if is_multi_instance else 'b n d -> b d'

    return Sequential(
        Reduce(pattern=ein_op, reduction=reduction),
        LayerNorm(normalized_shape=dim),
        Linear(dim, num_classes)
    )
