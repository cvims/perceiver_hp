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
import logging
from torch.utils.data import DataLoader
import torch.nn.functional as F
import src.utils as utils
from src.perceiver_pooling.model import PerceiverPooling
import torch
from typing import Dict
from torch.utils.tensorboard import SummaryWriter
from cvims.data.loaders import set_data_device, BeautifyDataLoaderIterations
from cvims.network.operator import NetworkOperator

# =============================================================================
# Helpers
# =============================================================================


def calculate_attention_weights(models: Dict[str, PerceiverPooling], device: torch.device, calc_mean: bool,
                                data_loaders: DataLoader or Dict[str, DataLoader]):
    all_data_loaders = copy.deepcopy(data_loaders)
    if isinstance(all_data_loaders, DataLoader):
        all_data_loaders = {m_name: copy.deepcopy(all_data_loaders) for m_name in models}

    all_attention_weights = {}

    for m_name in models:
        model = models[m_name]
        if not model.use_hopfield_pooling_fusion:
            logging.info('Skipping model attention weights calculation because hopfield pooling is turned off.')
            continue

        data_loader = all_data_loaders[m_name]

        attention_weights = {p: [] for p in model.perceivers}
        data_generator = BeautifyDataLoaderIterations(
            data_loader=data_loader,
            tqdm_description=''
        )

        for data in data_generator:
            batch, labels = data
            batch, _ = set_data_device(batch, labels, device=device)
            attention_weights_batch = model.get_attention_weights(data=batch)
            for m in attention_weights_batch:
                attention_weights[m].extend(attention_weights_batch[m])

        data_loader = None
        data_generator = None

        for p in attention_weights:
            attention_weights[p] = torch.stack(attention_weights[p])

        if not calc_mean:
            all_attention_weights[m_name] = attention_weights
            continue

        means = {}
        for perceiver_name in attention_weights:
            mean = None
            if attention_weights[perceiver_name] is not None:
                mean = torch.Tensor(attention_weights[perceiver_name]).mean(dim=0)
            means[perceiver_name] = mean

        all_attention_weights[m_name] = means

    return all_attention_weights


def additional_loss_callback(tensorboard_writer: SummaryWriter, mode: str, epoch: int, losses: torch.Tensor,
                             model_outputs: torch.Tensor, targets: torch.Tensor, **kwargs) -> None:
    """
    Method to calculate additional values after the training and eval iteration of operate. Adds the metrics to
    tensorboard if parameter is set to True.
    We calculate additional losses for the
    :param tensorboard_writer:
    :param mode:
    :param epoch:
    :param losses:
    :param model_outputs:
    :param targets:
    :return:
    """
    def calculate_perceiver_loss(operator: NetworkOperator, _device, _dataset, data_loader_kwargs):
        _model = operator.model

        separate_perceiver_models = utils.create_all_modality_perceivers(model=_model)

        # create individual data loader for separate perceiver models
        separate_model_data_loaders = utils.create_all_modality_perceivers_data_loader(
            perceiver_models=separate_perceiver_models, dataset=_dataset, **data_loader_kwargs
        )

        # we need to operator functionality to execute the predictions implementations
        separate_operators = utils.replicate_operator_with_sub_models(
            operator=operator, models=separate_perceiver_models
        )

        # Attention: This may take a while depending on your validation dataset size and amount of different modalities
        # Every single modality perceiver gets traversed with the validation dataset and plotted to tensorboard!
        # run predictions
        results = utils.run_predictions(
            operators=separate_operators,
            data_loaders=separate_model_data_loaders,
            device=_device,
            loss_fn=F.cross_entropy,
            calc_accuracy=True,
            acc_fn=utils.calculate_accuracy
        )

        _losses = {p_name: results[p_name]['avg_loss'] for p_name in results}
        _labels = [results[p_name]['targets'] for p_name in results]
        _accuracies = {p_name: results[p_name]['accuracy'] for p_name in results}

        return _losses, _accuracies, _labels

    if mode == 'eval' or mode == 'train':
        # get model
        current_operator = kwargs['operator']
        current_operator.model.eval()

        dataset = kwargs['train_dataset'] if mode == 'train' else kwargs['val_dataset']
        device = kwargs['device']
        dl_kwargs = {}
        if 'batch_size' in kwargs:
            dl_kwargs['batch_size'] = kwargs['batch_size']
        if 'persistent_workers' in kwargs:
            dl_kwargs['persistent_workers'] = kwargs['persistent_workers']
        if 'workers' in kwargs:
            dl_kwargs['workers'] = kwargs['workers']
        if 'pin_memory' in kwargs:
            dl_kwargs['pin_memory'] = kwargs['pin_memory']

        losses, accuracies, _ = calculate_perceiver_loss(
            operator=current_operator, _dataset=dataset, _device=device, data_loader_kwargs=dl_kwargs
        )

        # each perceivers loss and accuracy
        for perceiver_name in losses:
            tensorboard_writer.add_scalar('/'.join(['Loss', perceiver_name, mode]),
                                          losses[perceiver_name], epoch)
            tensorboard_writer.add_scalar('/'.join(['Accuracy', perceiver_name, mode]),
                                          accuracies[perceiver_name], epoch)

    # overall accuracy
    tensorboard_writer.add_scalar('/'.join(['Accuracy', mode]), utils.calculate_accuracy(model_outputs=model_outputs, targets=targets), epoch)
