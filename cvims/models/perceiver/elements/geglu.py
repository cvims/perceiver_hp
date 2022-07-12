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
from torch.nn import Module
import torch.nn.functional as func


__all__ = [
    'GEGLU'
]


class GEGLU(Module):
    """
    Source: https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
    URL date: 15/10/2021
    """
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * func.gelu(gates)
