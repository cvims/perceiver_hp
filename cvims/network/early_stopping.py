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
Early stopping implementation
"""
# =============================================================================
# Imports
# =============================================================================
import abc
import torch
from typing import List, Tuple


__all__ = [
    'CustomEarlyStopping', 'EarlyStoppingByLoss', 'AccuracyEarlyStopping'
]


class CustomEarlyStopping(metaclass=abc.ABCMeta):
    def __init__(self, patience: int, delta: float, optimization: str) -> None:
        """
        Early stopping interface as default base for new early stopping mechanisms.
        :param patience: Epochs to wait until firing to stop
        :param delta: Difference threshold to whether use the following epoch as improvement or not
        :param optimization: String value ('minimize' or 'maximize'). Optimization direction.
        """
        assert patience > 0
        self.patience = patience
        assert 0.0 < delta <= 1.0
        self.delta = delta
        assert optimization == 'maximize' or optimization == 'minimize'
        self.optimization = optimization
        self.best_score = None
        self.patience_count = 0

        # Variable to call to get the early stop state
        self.early_stop = False

    def reset_stopping_scores(self):
        """
        Reset previously calculated values for early stopping
        :return:
        """
        self.best_score = None
        self.patience_count = 0
        self.early_stop = False

    @staticmethod
    def _preprocess_calculate_score_inputs(
            losses: torch.Tensor or List[float],
            model_outputs: List[torch.Tensor] or List[float],
            targets: List[torch.Tensor] or List[float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param losses: Model losses
        :param model_outputs: Outputs of the network
        :param targets: Corresponding targets to predictions
        :return:
        """
        if isinstance(losses, list):
            losses = torch.Tensor(losses)
        if isinstance(model_outputs, list):
            model_outputs = torch.Tensor(model_outputs)
        if isinstance(targets, list):
            targets = torch.Tensor(targets)

        return losses, model_outputs, targets

    @abc.abstractmethod
    def _calculate_score(self, losses: torch.Tensor or List[float],
                         model_outputs: List[torch.Tensor] or List[float],
                         targets: List[torch.Tensor] or List[float]) -> float:
        """
        Method to implement to calculate to early stopping score.
        :param losses: Losses list of tensors
        :param model_outputs:
        :param targets:
        :return:
        """
        raise NotImplementedError

    def __call__(self, losses: torch.Tensor or List[float],
                 model_outputs: List[torch.Tensor] or List[float],
                 targets: List[torch.Tensor] or List[float]) -> None:
        """
        Early stopping routine
        :param losses: Model losses
        :param model_outputs: Outputs of the network
        :param targets: Corresponding targets to predictions
        :return:
        """
        calculated_score = self._calculate_score(
            losses=losses, model_outputs=model_outputs, targets=targets
        )

        if self.best_score is None:
            self.best_score = calculated_score
            return

        if self.optimization == 'maximize':
            if calculated_score > self.best_score + self.delta:
                self.best_score = calculated_score
                self.patience_count = 0
            else:
                self.patience_count += 1
        elif self.optimization == 'minimize':
            if calculated_score < self.best_score - self.delta:
                self.best_score = calculated_score
                self.patience_count = 0
            else:
                self.patience_count += 1

        if self.patience_count > self.patience:
            self.early_stop = True
        else:
            self.early_stop = False


class EarlyStoppingByLoss(CustomEarlyStopping):
    def __init__(self, patience: int, delta: float = 0.0) -> None:
        """
        Early stopping mechanism by using the loss values
        :param patience:
        :param delta:
        """
        super(EarlyStoppingByLoss, self).__init__(patience=patience, delta=delta, optimization='minimize')

    def _calculate_score(self, losses: torch.Tensor or List[float],
                         model_outputs: List[torch.Tensor] or List[float],
                         targets: List[torch.Tensor] or List[float]) -> float:
        """
        :param losses: Model losses
        :param model_outputs: Outputs of the network
        :param targets: Corresponding targets to predictions
        :return:
        """
        losses, model_outputs, targets = self._preprocess_calculate_score_inputs(
            losses=losses, model_outputs=model_outputs, targets=targets
        )

        avg_loss = torch.sum(losses) / torch.flatten(losses).size()[0]
        return avg_loss


class AccuracyEarlyStopping(CustomEarlyStopping):
    def __init__(self, patience: int, delta: float = 0.0) -> None:
        """
        Early stopping mechanism by using the accuracy metric
        :param patience:
        :param delta:
        """
        super(AccuracyEarlyStopping, self).__init__(patience=patience, delta=delta, optimization='maximize')

    def _calculate_score(self, losses: torch.Tensor or List[float],
                         model_outputs: torch.Tensor or List[float],
                         targets: torch.Tensor or List[float]) -> float:
        """
        :param losses: Model losses
        :param model_outputs: Outputs of the network
        :param targets: Corresponding targets to predictions
        :return:
        """
        losses, model_outputs, targets = self._preprocess_calculate_score_inputs(
            losses=losses, model_outputs=model_outputs, targets=targets
        )

        max_value_indices = torch.max(model_outputs, dim=1).indices
        accuracy = (max_value_indices == targets).to(dtype=torch.float32).mean()

        return accuracy
