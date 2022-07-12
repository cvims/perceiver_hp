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
Implementation of a network operator which use any kind of model and wraps it
into predefined functions.
"""
# =============================================================================
# Imports
# =============================================================================
import copy
import math
import time
import os
from abc import ABC
import torch.nn
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Any, Iterable, Callable, Iterator, Dict
from cvims.data.loaders import set_data_device
from cvims.network.early_stopping import CustomEarlyStopping


__all__ = [
    'NetworkOperator', 'build_timestamp_log_dir', 'load_model', 'save_model_information', 'save_model'
]


def build_timestamp_log_dir(log_dir: str) -> str:
    """
    Creates a default log dir based on timestamp
    :param log_dir: Base directory
    :return:
    """
    # format: day month year hour minute second
    time_obj = time.localtime(time.time())
    timestamp = '%d%d%d_%d%d%d' % (time_obj.tm_mday, time_obj.tm_mon, time_obj.tm_year,
                                   time_obj.tm_hour, time_obj.tm_min, time_obj.tm_sec)

    return os.path.join(log_dir, timestamp)


def load_model(model: torch.nn.Module, path: str, to_device: torch.device, model_state_key: str = None)\
        -> Tuple[torch.nn.Module, Any]:
    """
    Loads the torch model and other additional information.
    :param model: torch Module where to load the state dict into.
    :param path: full path (with filename) to the models state dictionary.
    :param to_device: torch.device to put the model
    :param model_state_key: Key of models load_state_dict to find the model specific weights
    :return:
    """
    assert os.path.isfile(path=path)

    model_info = torch.load(path, map_location=to_device)
    if model_state_key:
        model.load_state_dict(model_info[model_state_key])
    else:
        model.load_state_dict(model_info)

    model.eval()

    return model, model_info


def save_model_information(log_dir: str, file_name: str, information: Dict[str, Any]) -> None:
    """
    Saves the torch model and other information (kwargs) into the log_dir
    :param log_dir: Directory to save the model. The class defined log dir is used as directory.
    :param file_name: File name without extension.
    :param information: All additional information that should be stored (torch.save)
    :return:
    """

    # in case a file extension is provided
    file_name = file_name.split('.')[0]

    save_dir = os.path.join(log_dir)
    old_mask = os.umask(000)
    os.makedirs(save_dir, exist_ok=True)

    # save the information
    torch.save(information, os.path.join(save_dir, file_name + '.pth_info'))

    os.umask(old_mask)


def save_model(model: torch.nn.Module, log_dir: str, file_name: str, model_state_key: str, **kwargs) -> None:
    """
    Saves the torch model and other information (kwargs) into the log_dir
    :param model: torch model (nn.Module)
    :param log_dir: Directory to save the model. The class defined log dir is used as directory.
    :param model_state_key: Key of dict entry do differentiate the model from kwargs.
    :param file_name: File name without extension.
    :param kwargs: All additional information that should be stored (torch.save)
    :return:
    """
    info_dict = {
        model_state_key: model.state_dict(),
        **kwargs
    }

    # in case a file extension is provided
    file_name = file_name.split('.')[0]

    save_dir = os.path.join(log_dir)
    old_mask = os.umask(000)
    os.makedirs(save_dir, exist_ok=True)

    # save the information
    torch.save(info_dict, os.path.join(save_dir, file_name + '.pth'))

    os.umask(old_mask)


class NetworkOperator(ABC):
    def __init__(self, model: torch.nn.Module, model_name: str, log_dir: str or None) -> None:
        """
        Creates a network operator with predefined logging and operating functionality.
        :param model: torch model
        :param model_name: Representative torch model name
        :param log_dir: Logging directory
        """
        assert model is not None and isinstance(model, torch.nn.Module)
        self.model = model
        self.model_name = model_name

        self._state_dict_name = 'model_state_dict'

        # Once initialized it is fixed
        self._log_dir = None
        self.log_dir = log_dir

    @property
    def log_dir(self) -> str:
        """
        Global network operator log directory getter
        :return:
        """
        return self._log_dir

    @log_dir.setter
    def log_dir(self, path: str):
        """
        Global network operator log directory setter
        :param path: Log directory path
        :return:
        """
        if self._log_dir:
            # cannot be reinitialized
            return

        if path:
            self._log_dir = self._create_log_dir_path(log_dir=path)

    def overwrite_log_dir(self, path: str):
        """
        Only opportunity to overwrite a defined log dir
        :param path:
        :return:
        """
        self._log_dir = self._create_log_dir_path(log_dir=path)

    @staticmethod
    def _create_log_dir_path(log_dir: str) -> str:
        """
        Creates the path for the log directory
        :param log_dir: Base directory
        :return:
        """
        return build_timestamp_log_dir(log_dir=log_dir)

    def save_model(self, file_name: str, **kwargs) -> None:
        """
        Saves the torch model and other information (kwargs) into the log_dir
        :param file_name: File name without extension.
        :param kwargs: All additional information that should be stored (torch.save)
        :return:
        """
        assert self.log_dir, 'Provide a log dir for the network operator before starting the operation'

        save_model(
            model=self.model, log_dir=self.log_dir, file_name=file_name, model_state_key=self._state_dict_name, **kwargs
        )

    def load_model(self, file_name: str, to_device: torch.device)\
            -> Tuple[torch.nn.Module, Any]:
        """
        Loads the torch model and other additional information.
        :param file_name: File name without extension.
        :param to_device: torch.device to put the model
        :return:
        """
        assert self.log_dir, 'Provide a log dir for the network operator before starting the operation'

        return load_model(
            model=self.model, path=os.path.join(self.log_dir, file_name), model_state_key=self._state_dict_name,
            to_device=to_device
        )

    def _model_iteration(
            self, device: torch.device, data_loader: Iterable, optimizer: Optimizer = None,
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
            model_input_pre_hook: Callable[[Any, Any], Tuple[Any, Any]] = None,
            optimizer_pre_hook: Callable[[Iterator[Parameter]], Any] = None,
            after_batch_hook: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, ...], Any] = None,
            after_iterations_hook: Callable[..., Any] = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Model iteration implementation with different hook interactions.
        :param device: torch.device
        :param data_loader: Iterable object used for data looping. The first two objects from a iteration must be data
        and label where data is a torch.Tensor and label is the corresponding annotation (also torch.Tensor).
        :param optimizer: torch optimizer
        :param loss_fn: Loss function
        :param model_input_pre_hook: Hook to interact with data and label, respectively before inputting it to the
        network
        :param optimizer_pre_hook: Hook for analysing and updating the model parameters. Input gets parameters which are
        the model parameters. The function output is ignored.
        :param after_batch_hook: Hook for every batch iteration of the data loader. Gets called at the end of each
        iteration with inputs (losses, model outputs, targets, *args) where args are the outputs from the data loader
        :param after_iterations_hook: Hook at the end of the iteration. For example learning schedulers.
        :param kwargs: Kwargs for the model forward pass.
        :return:
        """
        losses, model_outputs, targets = [], [], []

        for data in data_loader:
            batch, labels, *other_data = data
            # set the device of the batch and labels
            batch, labels = set_data_device(data=batch, labels=labels, device=device)

            if model_input_pre_hook:
                batch, labels = model_input_pre_hook(batch, labels)

            # call the model forward
            model_output = self.model.forward(batch, **kwargs)

            _loss = None
            if loss_fn:
                # Calculate loss
                _loss = loss_fn(model_output, labels)

            if optimizer:
                # Update models parameters
                optimizer.zero_grad()

                # we only go backwards if the optimizer is set
                _loss.backward()

                # Call pre optimizer hook if there is one defined
                if optimizer_pre_hook:
                    optimizer_pre_hook(self.model.parameters())

                optimizer.step()

            # Collect targets and outputs and loss
            if _loss:
                losses.append(_loss.detach())
            _targets = labels.detach()
            targets.append(_targets)
            _model_outputs = model_output.detach()
            model_outputs.append(_model_outputs)

            # Call after batch hook
            if after_batch_hook:
                after_batch_hook(_loss, _targets, _model_outputs, *data)

        # make tensors out of losses, model outputs and targets
        losses = torch.stack(losses) if losses else None
        model_outputs = torch.cat(model_outputs, dim=0)
        targets = torch.cat(targets, dim=0)

        # Call after epoch hook
        if after_iterations_hook:
            after_iterations_hook()

        return losses, model_outputs, targets

    def train_epoch(
            self, device: torch.device, data_loader: Iterable, optimizer: Optimizer,
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            model_input_pre_hook: Callable[[Any, Any], Tuple[Any, Any]] = None,
            optimizer_pre_hook: Callable[[Iterator[Parameter]], Any] = None,
            after_batch_hook: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, ...], Any] = None,
            after_iteration_hook: Callable[..., Any] = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Train epoch implementation with different hook interactions.
        :param device: torch.device
        :param data_loader: Iterable object used for data looping. The first two objects from a iteration must be data
        and label where data is a torch.Tensor and label is the corresponding annotation (also torch.Tensor).
        :param optimizer: torch optimizer
        :param loss_fn: Loss function
        :param model_input_pre_hook: Hook to interact with data and label, respectively before inputting it to the
        network
        :param optimizer_pre_hook: Hook for analysing and updating the model parameters. Input gets parameters which are
        the model parameters. The function output is ignored.
        :param after_batch_hook: Hook for every batch iteration of the data loader. Gets called at the end of each
        iteration with inputs (losses, model outputs, targets, *args) where args are the outputs from the data loader
        :param after_iteration_hook: Hook at the end of the iteration. For example learning schedulers.
        :param kwargs: Kwargs for the model forward pass.
        :return:
        """
        # set training mode for model
        self.model.train()

        with torch.enable_grad():
            return self._model_iteration(
                device=device, data_loader=data_loader, optimizer=optimizer,
                model_input_pre_hook=model_input_pre_hook, optimizer_pre_hook=optimizer_pre_hook,
                loss_fn=loss_fn, after_batch_hook=after_batch_hook, after_iterations_hook=after_iteration_hook, **kwargs
            )

    def eval_iter(
            self, device: torch.device, data_loader: Iterable,
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
            model_input_pre_hook: Callable[[Any, Any], Tuple[Any, Any]] = None,
            after_batch_hook: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Any] = None,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Eval epoch implementation with different hook interactions.
        :param device: torch.device
        :param data_loader: Iterable object used for data looping. The first two objects from a iteration must be data
        and label where data is a torch.Tensor and label is the corresponding annotation (also torch.Tensor).
        :param loss_fn: Loss function
        :param model_input_pre_hook: Hook to interact with data and label, respectively before inputting it to the
        network
        :param after_batch_hook: Hook for every batch iteration of the data loader. Gets called at the end of each
        iteration with inputs (losses, model outputs, targets, *args) where args are the outputs from the data loader
        :param kwargs: Kwargs for the model forward pass.
        :return:
        """
        # set eval mode for model
        self.model.eval()

        with torch.no_grad():
            return self._model_iteration(
                device=device, data_loader=data_loader, optimizer=None,
                model_input_pre_hook=model_input_pre_hook, optimizer_pre_hook=None,
                loss_fn=loss_fn, after_batch_hook=after_batch_hook, after_iterations_hook=None, **kwargs
            )

    def operate(
            self, epochs: int, device: torch.device, train_data_loader: Iterable, eval_data_loader: Iterable,
            optimizer: Optimizer, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            model_input_pre_hook: Callable[[Any, Any], Tuple[Any, Any]] = None,
            optimizer_pre_hook: Callable[[Iterator[Parameter]], None] = None,
            train_after_batch_hook: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None] = None,
            eval_after_batch_hook: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None] = None,
            after_epoch_hook: Callable[[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                                        ], None] = None,
            early_stopping: CustomEarlyStopping = None,
            log_active: bool = True, use_tensorboard: bool = False, save_best_only: bool = False,
            tensorboard_hook: Callable[[SummaryWriter, str, int, torch.Tensor, torch.Tensor, torch.Tensor],
                                       None] = None,
            loss_optimization: str = 'minimize', **kwargs
    ) -> torch.nn.Module:
        """
        Combines train and eval and adds logging mechanisms and operations to it. This is a wrapper method for training.
        :param epochs: Epochs for training and evaluation.
        :param device: torch device
        :param train_data_loader: data loader for training iterations
        :param eval_data_loader: data loader for evaluation iterations
        :param optimizer: torch optimizer
        :param loss_fn: Loss function
        :param model_input_pre_hook: Hook to interact with data and label, respectively before inputting it to the
        network
        :param optimizer_pre_hook: Hook for analysing and updating the model parameters. Input gets parameters which are
            the model parameters. The function output is ignored.
        :param train_after_batch_hook: Hook for every batch iteration of the training data loader. Gets called at the
            end of each  iteration with inputs (losses, model outputs, targets, *args) where args are the outputs from
            the data loader
        :param eval_after_batch_hook: Hook for every batch iteration of the training data loader. Gets called at the
            end of each  iteration with inputs (losses, model outputs, targets, *args) where args are the outputs from
            the data loader
        :param after_epoch_hook: Hook at the end of every epoch, after train and eval execution.
            Contains two tuples as inputs which contain losses, model outputs and targets for train and eval,
            respectively.
        :param early_stopping: Early stopping (CustomEarlyStopping implementation)
        :param log_active: If False then 'use_tensorboard' and 'save_best_only' and all logging events are ignored
        :param use_tensorboard: Opens a tensorboard writer if true else no logging with tensorboard is applied.
        :param save_best_only: Saves only the best model if True else every epoch.
        :param tensorboard_hook: Callable for additionally writing into the tensorboard log.
            Parameters are (tensorboard_writer: SummaryWriter, mode: str, epoch: int, losses: tensor,
            model_outputs: tensor, targets: tensor).
            Parameter 'mode' is a string which contains either train (for training information) or eval (for evaluation
            information). Use tensorboard_writer object to add information to the tensorboard.
        :param loss_optimization: Optimization direction for deciding for best model. Possible values are 'minimize' or
            'maximize'
        :param kwargs: Kwargs for the model forward pass.
        :return: Best model (torch.nn.Module)
        """
        best_model = None
        loss_optimization = loss_optimization.lower()
        assert loss_optimization == 'minimize' or loss_optimization == 'maximize'

        if log_active:
            assert self.log_dir, 'Provide a log dir for the network operator before starting the operation'

        tensorboard_summary_writer = None
        if log_active and use_tensorboard:
            tensorboard_summary_writer = SummaryWriter(log_dir=self.log_dir)

        # for keeping track of best model
        prev_best_loss = math.inf if loss_optimization == 'minimize' else -math.inf

        for epoch in range(0, epochs):
            print('Epoch:', epoch)

            # model training
            t_losses, t_model_outputs, t_targets = self.train_epoch(
                device=device, data_loader=train_data_loader, optimizer=optimizer,
                optimizer_pre_hook=optimizer_pre_hook, loss_fn=loss_fn,
                model_input_pre_hook=model_input_pre_hook,
                after_batch_hook=train_after_batch_hook, after_iteration_hook=None, **kwargs
            )
            t_avg_loss = torch.mean(t_losses)

            # model evaluation
            e_losses, e_model_outputs, e_targets = self.eval_iter(
                device=device, data_loader=eval_data_loader, loss_fn=loss_fn,
                model_input_pre_hook=model_input_pre_hook,
                after_batch_hook=eval_after_batch_hook, **kwargs
            )
            e_avg_loss = torch.mean(e_losses)

            # tensorboard interaction
            if use_tensorboard and tensorboard_summary_writer:
                # add known train and eval information
                tensorboard_summary_writer.add_scalar('Loss/train', t_avg_loss, epoch)
                tensorboard_summary_writer.add_scalar('Loss/val', e_avg_loss, epoch)

                # check if tensorboard hook exists
                if tensorboard_hook:
                    # for train information
                    tensorboard_hook(tensorboard_summary_writer, 'train', epoch, t_losses, t_model_outputs, t_targets)
                    # for eval information
                    tensorboard_hook(tensorboard_summary_writer, 'eval', epoch, e_losses, e_model_outputs, e_targets)

            # save best model
            if (loss_optimization == 'minimize' and prev_best_loss > e_avg_loss) or \
                    (loss_optimization == 'maximize' and prev_best_loss < e_avg_loss):
                prev_best_loss = e_avg_loss
                best_model = copy.deepcopy(self.model)

            if log_active:
                self.save_model(
                    file_name='best_model.pth'.format(epoch=epoch)
                )

                # save for each epoch too
                if not save_best_only:
                    self.save_model(
                        file_name='model_epoch_{epoch}.pth'.format(epoch=epoch)
                    )

            # call the hook if set
            if after_epoch_hook:
                after_epoch_hook(
                    (t_losses, t_model_outputs, t_targets), (e_losses, e_model_outputs, e_targets)
                )

            # early stopping
            if early_stopping:
                early_stopping(losses=e_losses, model_outputs=e_model_outputs, targets=e_targets)

                if early_stopping.early_stop:
                    print(f'Early stopping at epoch {epoch}')
                    break

        # close the tensorboard writer
        if tensorboard_summary_writer:
            tensorboard_summary_writer.close()

        return best_model
