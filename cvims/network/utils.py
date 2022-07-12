#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Organization  : Technische Hochschule Ingolstadt
# Institution   : Computer Vision for Intelligent Mobility Systems (CVIMS)
# Created By    : Dominik Rößle
# Created Date  : Tuesday April 12 hh:mm:ss GMT 2022
# Latest Update : Tuesday April 12 hh:mm:ss GMT 2022
# =============================================================================
"""
Network utils
"""
# =============================================================================
# Imports
# =============================================================================
import torch


__all__ = ['get_model_trainable_parameter_count', 'get_model_component_trainable_parameter_count']


def get_model_trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_component_trainable_parameter_count(model: torch.nn.Module) -> dict:
    count_container = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        section = name.split('.')[0]
        if section in count_container:
            count_container[section] = count_container[section] + p.numel()
        else:
            count_container[section] = p.numel()
    return count_container
