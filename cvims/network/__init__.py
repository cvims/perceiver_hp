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
Init file for network package.
Contains network related operations, e. g. early stopping
"""
# =============================================================================
# Imports
# =============================================================================
from .early_stopping import CustomEarlyStopping, EarlyStoppingByLoss, AccuracyEarlyStopping
from .operator import NetworkOperator
from .utils import get_model_trainable_parameter_count, get_model_component_trainable_parameter_count

__all__ = [
    'CustomEarlyStopping', 'EarlyStoppingByLoss', 'AccuracyEarlyStopping',
    'NetworkOperator',
    'get_model_trainable_parameter_count', 'get_model_component_trainable_parameter_count'
]
