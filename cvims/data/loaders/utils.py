#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Organization  : Technische Hochschule Ingolstadt
# Institution   : Computer Vision for Intelligent Mobility Systems (CVIMS)
# Created By    : Dominik Rößle
# =============================================================================
"""
Utils for torch data loaders
"""
# =============================================================================
# Imports
# =============================================================================
import os
from collections.abc import Iterable

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple


__all__ = [
    'create_data_loader', 'set_data_device', 'BeautifyDataLoaderIterations'
]


def create_data_loader(dataset: Dataset, workers: int, batch_size: int, **kwargs) -> DataLoader:
    """
    Function to create a train and test data loader.
    """
    assert workers >= 1 and batch_size >= 1
    workers = min(os.cpu_count(), workers)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, **kwargs)

    return data_loader


def set_data_device(data: Dict[str, torch.Tensor] or List[torch.Tensor] or torch.Tensor or None,
                    labels: Dict[str, torch.Tensor] or List[torch.Tensor] or torch.Tensor or None,
                    device: torch.device) \
        -> Tuple[Dict[str, torch.Tensor] or List[torch.Tensor] or torch.Tensor,
                 Dict[str, torch.Tensor] or List[torch.Tensor] or torch.Tensor]:
    """
    Sets the data to the torch device of your choice for different formats (dict, list, or tensors)
    :param data: Data to set the device
    :param labels: Corresponding labels to set the device
    :param device: torch.device
    :return: data, label of same types as the input parameters
    """
    # set device for data
    if data is not None:
        if isinstance(data, dict):
            # iterate all data elements
            for data_name in data:
                if data[data_name] is not None:
                    data[data_name] = data[data_name].to(device=device)
        elif isinstance(data, list):
            for i, d in enumerate(data):
                data[i] = d.to(device=device)
        else:
            data = data.to(device=device)

    # set device for labels
    if labels is not None:
        if labels is not None:
            if isinstance(labels, dict):
                # iterate all data elements
                for label_name in labels:
                    if labels[label_name] is not None:
                        labels[label_name] = labels[label_name].to(device=device)
            elif isinstance(labels, list):
                for i, l in enumerate(labels):
                    labels[i] = l.to(device=device)
            else:
                labels = labels.to(device=device)

    return data, labels


class BeautifyDataLoaderIterations(Iterable):
    def __init__(self, data_loader: DataLoader, tqdm_description: str = None, tqdm_unit: str = 'batch'):
        """
        Uses tqdm visualizations for dataset iterations.
        :param data_loader: torch data loader
        :param tqdm_description: Visualization text
        :param tqdm_unit: tqdm unit description
        """
        self.data_loader = data_loader
        self._org_next_data = None
        self.tqdm_description = tqdm_description
        self.tqdm_unit = tqdm_unit
        self.tqdm_num = 0
        self.tqdm_bar = None

    def _beautify(self):
        self.tqdm_bar = tqdm(
            desc=self.tqdm_description, total=self.data_loader.__len__(), unit=self.tqdm_unit
        )

    def _next_data(self):
        if self.tqdm_num >= self.data_loader.__len__():
            self.tqdm_num = 0
            self.tqdm_bar.close()
            self.tqdm_bar = None
        else:
            self.tqdm_num += 1
            self.tqdm_bar.update(1)

        next_data = self._org_next_data()

        return next_data

    def __iter__(self):
        # open a new tqdm bar
        self._beautify()
        iter_obj = self.data_loader.__iter__()
        if self._org_next_data is None:
            self._org_next_data = iter_obj._next_data
        iter_obj._next_data = self._next_data
        return iter_obj
