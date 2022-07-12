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
from torch.nn import Sequential, Linear, Dropout, Module
from .geglu import GEGLU


__all__ = [
    'FeedForward'
]


class FeedForward(Module):
    """"
    Source: https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
    URL date: 15/10/2021
    """
    def __init__(self, dim, mult=1, dropout=0.0):
        super().__init__()
        self.net = Sequential(
            Linear(dim, dim * mult * 2),
            GEGLU(),
            Linear(dim * mult, dim),
            Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
