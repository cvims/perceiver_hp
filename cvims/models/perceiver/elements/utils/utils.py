#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Organization  : Technische Hochschule Ingolstadt
# Institution   : Computer Vision for Intelligent Mobility Systems (CVIMS)
# Created By    : Dominik Rößle
# Created Date  : DayName Month Day hh:mm:ss GMT YYYY
# Latest Update : DayName Month Day hh:mm:ss GMT YYYY
# =============================================================================
"""
File description.
"""
# =============================================================================
# Imports
# =============================================================================
from typing import Any

import torch
from math import pi


__all__ = [
    'exists', 'default', 'fourier_encode'
]


def exists(val: Any):
    return val is not None


def default(val: Any, d: Any):
    return val if exists(val) else d


def fourier_encode(x: torch.Tensor, max_freq: float, num_bands: int):
    """
    Source: https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
    URL date: 10/03/2022
    :param x:
    :param max_freq:
    :param num_bands:
    :return:
    """
    assert num_bands > 0, 'Please provide a number of bands greater 0 to generate fourier encodings'
    x = x.unsqueeze(-1)
    device, d_type, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=d_type)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)

    return x
