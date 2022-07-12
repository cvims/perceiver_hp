#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Organization  : Technische Hochschule Ingolstadt
# Institution   : Computer Vision for Intelligent Mobility Systems (CVIMS)
# Created By    : Dominik Rößle
# =============================================================================
"""
File description.
"""
# =============================================================================
# Imports
# =============================================================================
import copy
import os
from typing import Dict, Callable, List
import torch
from torch.nn import ModuleDict
from torch.utils.data import DataLoader
from cvims.data.loaders import create_data_loader
from cvims.data.transforms.cut import TransformModalityCut
from cvims.data.transforms.utils import TransformsCompose
from cvims.network.operator import NetworkOperator
from src.datasets import DynamicDatasetInterface
from src.perceiver_pooling.model import PerceiverPooling


def calculate_accuracy(model_outputs: torch.Tensor or List, targets: torch.Tensor or List, calc_mean: bool = True) \
        -> torch.Tensor:
    """
    Uses models outputs and targets to calculate the accuracy
    :param model_outputs: Model outputs
    :param targets: Target labels
    :param calc_mean: If True the accuracy is one number (as usual) if False then the return is a tensor of
    model output and target comparisons
    :return:
    """
    # accuracy
    # stacking at dim 0 only available if network batch size output is permanently equal
    model_outputs = torch.cat(model_outputs, dim=0) if isinstance(model_outputs, list) else model_outputs
    targets = torch.cat(targets, dim=0) if isinstance(targets, list) else targets
    max_value_indices = torch.max(model_outputs, dim=1).indices
    if calc_mean:
        return (max_value_indices == targets).to(dtype=torch.float32).mean().detach()
    else:
        return (max_value_indices == targets).detach()


def save_model_configs(log_dir, main_config) -> None:
    """
    :param log_dir: Logging directory
    :param main_config: Main config file (which includes the names of perceiver models)
    """
    # use initialized log dir to write config file
    os.makedirs(log_dir, exist_ok=False)
    # change root path of config
    root_config = main_config['root_config']
    root_config['PERCEIVERS']['PERCEIVER_CONFIG_PATH'] = '.'
    with open(os.path.join(log_dir, 'config.ini'), 'w') as configfile:  # save
        root_config.write(configfile)
    
    perceiver_params = main_config['perceivers']

    # save perceiver configs
    for perceiver_config_name in list(perceiver_params.keys()):
        with open(os.path.join(log_dir, perceiver_config_name + '.ini'), 'w') as perceiver_conf_file:
            config_parser = perceiver_params[perceiver_config_name]['config']
            config_parser.write(perceiver_conf_file)


def replicate_operator_with_sub_models(operator: NetworkOperator, models: ModuleDict) -> Dict[str, NetworkOperator]:
    """
    Replicates the network operator with new sub models (specific for perceiver dynamic model executions)
    :param operator:
    :param models:
    :return:
    """
    operators = {}
    for model in models:
        _operator = copy.deepcopy(operator)
        _operator.model = models[model]
        operators[model] = _operator

    return operators


def create_all_modality_perceivers(model: PerceiverPooling, training: bool = False) -> ModuleDict:
    """
    Uses the whole model and creates the architecture for each individual perceiver input pipe.
    Fusion and output layers are added for each individual perceiver
    :param model:
    :param training:
    :return:
    """
    all_modality_perceivers = ModuleDict({})

    for perceiver_name in model.perceivers:
        cut_model = model.create_individual_perceiver_model(
            perceiver_key=perceiver_name, keep_weights=True
        )
        if training:
            cut_model.train()
        else:
            cut_model.eval()
        all_modality_perceivers.add_module(perceiver_name, cut_model)

    if training:
        all_modality_perceivers.train()
    else:
        all_modality_perceivers.eval()

    return all_modality_perceivers


def create_all_modality_perceivers_data_loader(perceiver_models, dataset: DynamicDatasetInterface,
                                               **data_loader_kwargs) -> Dict[str, DataLoader]:
    """
    Creates individual data loaders for separate perceiver models
    """
    separate_model_data_loaders = {}
    for perceiver_name in perceiver_models:
        c_dataset = copy.deepcopy(dataset)
        restricted_modalities = [name for name in perceiver_models if name != perceiver_name]
        transform_add = TransformModalityCut(restricted_modalities=restricted_modalities)
        if c_dataset.transform:
            c_dataset.transform = TransformsCompose(transform=[transform_add, c_dataset.transform])
        else:
            c_dataset.transform = transform_add

        workers = 1
        if data_loader_kwargs and 'workers' in data_loader_kwargs:
            workers = data_loader_kwargs['workers']
            del data_loader_kwargs['workers']

        separate_model_data_loaders[perceiver_name] = create_data_loader(
            dataset=c_dataset, workers=workers, **data_loader_kwargs
        )

    return separate_model_data_loaders


def run_prediction(operator: NetworkOperator, data_loader: DataLoader, device: torch.device, loss_fn=None,
                   calc_accuracy: bool = True, acc_fn: Callable = None) -> Dict[str, torch.Tensor]:
    """
    Uses the operators eval_iter to predict the model outputs by using the given data loader.
    """
    if calc_accuracy:
        assert isinstance(acc_fn, Callable)

    # run the whole model (predict)
    losses, model_outputs, targets = operator.eval_iter(
        device=device,
        data_loader=data_loader,
        loss_fn=loss_fn
    )

    results = {
        'avg_loss': torch.mean(losses),
        'losses': losses,
        'model_outputs': model_outputs,
        'targets': targets,
    }

    if calc_accuracy:
        results.update({'accuracy': acc_fn(model_outputs=model_outputs, targets=targets)})

    return results


def run_predictions(operators: Dict[str, NetworkOperator], data_loaders: DataLoader or Dict[str, DataLoader],
                    device: torch.device, loss_fn=None, calc_accuracy: bool = True, acc_fn: Callable = None) -> Dict:
    """
    Uses each operators eval_iter to predict the model outputs by using the given data loader.
    """
    if calc_accuracy:
        assert isinstance(acc_fn, Callable)

    results = {}
    all_data_loaders = data_loaders
    if isinstance(data_loaders, DataLoader):
        all_data_loaders = {m_name: data_loaders for m_name in operators}

    for name in operators:
        result = run_prediction(
            operator=operators[name],
            device=device,
            data_loader=all_data_loaders[name],
            loss_fn=loss_fn,
            calc_accuracy=calc_accuracy,
            acc_fn=acc_fn
        )

        results[name] = result

    return results
