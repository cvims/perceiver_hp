#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Organization  : Technische Hochschule Ingolstadt
# Institution   : Computer Vision for Intelligent Mobility Systems (CVIMS)
# Created By    : Dominik Rößle
# =============================================================================
"""
Implementation for cut operations (multi-instance and multi-modality)
"""
# =============================================================================
# Imports
# =============================================================================
import torch
from typing import List, Dict


__all__ = [
    'TransformModalityCut', 'TransformMultiInstanceInstanceCut', 'TransformMultiModalMultiInstanceInstanceCut'
]


class TransformModalityCut(torch.nn.Module):
    def __init__(self, restricted_modalities: str or List[str]):
        """
        Cuts modalities by their name. Data of modalities is deleted completely from the passed data dict.
        :param restricted_modalities: Name(s) of modalities to restrict to pass
        """
        super(TransformModalityCut, self).__init__()
        if isinstance(restricted_modalities, str):
            self.restricted_modalities = [restricted_modalities]
        else:
            self.restricted_modalities = restricted_modalities

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        keys = list(data.keys())
        for key in keys:
            if key in self.restricted_modalities:
                del data[key]
        return data


class TransformMultiInstanceInstanceCut(torch.nn.Module):
    def __init__(self, cut_len: int):
        """
        Cuts instances of a multi instance dataset by a defined len
        :param cut_len: Max amount of instances to allow to pass
        """
        super(TransformMultiInstanceInstanceCut, self).__init__()
        self.cut_len = cut_len

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        max_instances = min(data.size()[0], self.cut_len)
        return data[:max_instances]


class TransformMultiModalMultiInstanceInstanceCut(TransformMultiInstanceInstanceCut):
    def __init__(self, cut_len: int, restricted_modalities: str or List[str] = None):
        """
        Cuts instances of a multi modal multi instance dataset by a defined len
        :param cut_len: Max amount of instances to allow to pass
        :param restricted_modalities: Restrict certain modalities to get transformed
        """
        super(TransformMultiModalMultiInstanceInstanceCut, self).__init__(
            cut_len=cut_len
        )
        self.restricted_modalities = restricted_modalities
        if not self.restricted_modalities:
            self.restricted_modalities = []

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for name in data:
            if name not in self.restricted_modalities:
                data[name] = super().forward(data=data[name])

        return data
