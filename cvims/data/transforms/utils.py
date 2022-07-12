#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Organization  : Technische Hochschule Ingolstadt
# Institution   : Computer Vision for Intelligent Mobility Systems (CVIMS)
# Created By    : Dominik Rößle
# =============================================================================
"""
Utils of transform methods of single instances (usual learning), multi-instance, multi-modal and
multi-modal multi-instance in combination.
"""
# =============================================================================
# Imports
# =============================================================================
import torch
from typing import Callable, Any, List, Dict, Tuple


__all__ = [
    'normalize',
    'SingleInstanceTransforms',
    'MultiInstanceTransforms',
    'MultiModalTransforms',
    'MultiModalMultiInstanceTransforms',
    'TransformsCompose'
]


def normalize(tensor: torch.Tensor, minimum: float or torch.Tensor, maximum: float or torch.Tensor)\
        -> Tuple[torch.Tensor, float, float]:
    """
    Normalizes the whole tensor by passed minimum and maximum. If you pass a batch of tensors then the normalization
    takes place over the whole batch and not instance-wise.
    :param tensor: torch.Tensor
    :param minimum: Min value for normalization
    :param maximum: Max value for normalization
    :return:
    """
    assert minimum != maximum

    if minimum > maximum:
        if isinstance(minimum, torch.Tensor):
            _maximum = torch.clone(minimum)
        else:
            _maximum = minimum

        minimum = maximum
        maximum = _maximum

    c_tensor = torch.clone(tensor)

    min_val, max_val = c_tensor.min(), c_tensor.max()
    if min_val != max_val:
        _range = max_val - min_val
    else:
        _range = 1

    new_range = maximum - minimum

    new_tensor = new_range * (c_tensor - min_val) / _range + minimum

    return new_tensor, min_val, max_val


class SingleInstanceTransforms(torch.nn.Module):
    """
    Interface for CVIMS custom transform methods
    """
    def __init__(self, probability: float, inplace: bool = False, *args, **kwargs) -> None:
        """
        Defines the single instance custom transforms module
        :param probability: Probability of using transforms
        :param inplace: Perform on tensor if True otherwise the tensor gets cloned
        """
        assert 0.0 < probability <= 1.0
        super(SingleInstanceTransforms, self).__init__()
        self.probability = probability

        # set inplace always True and control data (tensor) clones from this class (_call_impl)
        self._inplace = inplace

    def _call_impl(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self.inplace = True

        if torch.rand(size=(1,)) <= self.probability:
            if not self._inplace:
                data = torch.clone(data)
            return super().__call__(data, *args, **kwargs)

        # we only return data
        return data

    __call__: Callable[..., Any] = _call_impl


class MultiInstanceTransforms(SingleInstanceTransforms):
    def __init__(self, individual_instance_probabilities: bool, instance_probability: float,
                 apply_instances_as_batch: bool, inplace: bool = False, *args, **kwargs) -> None:
        """
        Defines the multi instance custom transforms module
        :param individual_instance_probabilities: Individual probabilities for instances (True / False).
        If False then either all instances or no instance is transformed.
        :param instance_probability: Probability of applying transform method
        :param apply_instances_as_batch: Applies all instances as one tensor if True or
        each instance separately if False
        :param inplace: Perform on tensor if True otherwise the tensor gets cloned
        if False
        """
        # we set inplace always to True and handle it separately here
        super(MultiInstanceTransforms, self).__init__(
            *args, **{'probability': instance_probability,
                      'individual_instance_probabilities': individual_instance_probabilities,
                      'instance_probability': instance_probability, 'inplace': inplace,
                      'apply_instances_as_batch': apply_instances_as_batch, **kwargs}
        )
        self._inplace = inplace
        self.individual_instance_probabilities = individual_instance_probabilities
        self.apply_instances_as_batch = apply_instances_as_batch

    def _call_impl(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self.inplace = True
        _initial_inplace = self._inplace

        # store initial probability
        initial_probability = self.probability

        if not self.individual_instance_probabilities:
            if torch.rand(size=(1,)) > self.probability:
                # we return only the first value as it must represent the data input
                return data
            else:
                # set probability to 1 for all instances
                self.probability = 1.0

        if not self._inplace:
            data = torch.clone(data)
            # for other transforms calls (inheritance)
            self._inplace = True

        # call super method (which calls forward method with passed parameters
        if not self.apply_instances_as_batch:
            for i, instance in enumerate(data):
                data[i] = super().__call__(data=data[i], *args, **kwargs)
        else:
            data = super().__call__(data=data, *args, **kwargs)

        # reset initial probability
        self.probability = initial_probability

        # reset initial _inplace
        self._inplace = _initial_inplace

        # return forward result
        return data

    __call__: Callable[..., Any] = _call_impl


class MultiModalTransforms(SingleInstanceTransforms):
    def __init__(self, individual_modality_probabilities: bool, modality_probability: float,
                 restricted_modalities: str or List[str] = None, inplace: bool = False, *args, **kwargs) -> None:
        """
        Defines the multi instance custom transforms module
        :param individual_modality_probabilities: Individual probabilities for instances (True / False).
        If False then either all instances or no instance is transformed
        :param modality_probability: Probability of applying transform method
        :param restricted_modalities: String value or List of strings with the names of the modalities which should not
        be transformed. None or empty string results in all modalities accepted otherwise the passed names will
         be restricted
        :param inplace: Perform on tensor if True otherwise the tensor gets cloned first
        """
        # we set inplace always to True and handle it separately here
        super(MultiModalTransforms, self).__init__(
            *args, **{'probability': modality_probability,
                      'individual_modality_probabilities': individual_modality_probabilities,
                      'modality_probability': modality_probability,
                      'restricted_modalities': restricted_modalities, 'inplace': inplace, **kwargs}
        )
        self._inplace = inplace
        self.individual_modality_probabilities = individual_modality_probabilities
        if restricted_modalities is not None:
            assert isinstance(restricted_modalities, str) or isinstance(restricted_modalities, list)

        if restricted_modalities is None:
            restricted_modalities = []
        elif isinstance(restricted_modalities, str):
            restricted_modalities = [restricted_modalities]

        self.restricted_modalities = []
        for restricted_modality in restricted_modalities:
            self.restricted_modalities.append(restricted_modality.lower())

    def _call_impl(self, data: Dict[str, torch.Tensor], *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Gets called before forward function
        :param data: Dictionary of modality name and tensor
        :param args:
        :param kwargs:
        :return:
        """
        self.inplace = True
        _initial_inplace = self._inplace

        # store initial probability
        initial_probability = self.probability

        if not self.individual_modality_probabilities:
            if torch.rand(size=(1,)) > self.probability:
                # we return only the first value as it must represent the data input
                return data
            else:
                # set probability to 1 for all instances
                self.probability = 1.0

        # call super method (which calls forward method with passed parameters
        # always modality by modality
        for modality in data:
            # go to next iteration if modality is restricted to transform
            if modality.lower() in self.restricted_modalities:
                continue

            if not self._inplace:
                data[modality] = torch.clone(data[modality])

            data[modality] = super().__call__(data=data[modality], *args, **kwargs)

        # reset initial probability
        self.probability = initial_probability

        # reset initial _inplace
        self._inplace = _initial_inplace

        # return forward result
        return data

    __call__: Callable[..., Any] = _call_impl


class MultiModalMultiInstanceTransforms(MultiModalTransforms, MultiInstanceTransforms):
    def __init__(self, individual_modality_probabilities: bool, individual_instance_probabilities: bool,
                 modality_probability: float, instance_probability: float,
                 restricted_modalities: str or List[str] = None, inplace: bool = False,
                 apply_instances_as_batch: bool = False, *args, **kwargs) -> None:
        """
        Defines the multi modal multi instance custom transforms
        :param individual_modality_probabilities: Use individual probabilities for modalities if True. Use either
        all or no modalities if False. Both cases are based on the modality probability.
        :param individual_instance_probabilities: Use individual probabilities for instances if True. Use either
        all or no instances if False. Both cases are based on the instance probability.
        :param modality_probability: Probability of applying transform method to modalities.
        :param instance_probability: Probability of applying transform method to instances.
        :param restricted_modalities: Restrict certain modalities to get transformed.
        :param inplace: Perform on tensor if True otherwise the tensor gets cloned
        :param apply_instances_as_batch: Applies transforms as one tensor if True or each instance separately if False
        :param args: args
        :param kwargs: kwargs
        """
        super(MultiModalMultiInstanceTransforms, self).__init__(
            *args,
            individual_modality_probabilities=individual_modality_probabilities,
            individual_instance_probabilities=individual_instance_probabilities,
            modality_probability=modality_probability, instance_probability=instance_probability,
            restricted_modalities=restricted_modalities, apply_instances_as_batch=apply_instances_as_batch,
            inplace=inplace,
            **kwargs
        )

        # probability is a SingleInstanceTransforms attribute
        self.probability = instance_probability

    def _call_impl(self, data: Dict[str, torch.Tensor], *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Gets called before forward function
        :param data: Dictionary of modality name and tensor
        :param args:
        :param kwargs:
        :return:
        """
        self.inplace = True
        _initial_inplace = self._inplace

        # store initial probability
        initial_probability = self.probability

        if not self.individual_modality_probabilities:
            if torch.rand(size=(1,)) > self.probability:
                # we return only the first value as it must represent the data input
                return data
            else:
                # set probability to 1 for all instances
                self.probability = 1.0

        # call super method (which calls forward method with passed parameters
        # always modality by modality
        for modality in data:
            # go to next iteration if modality is restricted to transform
            if modality.lower() in self.restricted_modalities:
                continue

            if not self._inplace:
                data[modality] = torch.clone(data[modality])

            data[modality] = MultiInstanceTransforms._call_impl(self=self, data=data[modality], *args, **kwargs)

        # reset initial probability
        self.probability = initial_probability

        # reset initial _inplace
        self._inplace = _initial_inplace

        # return forward result
        return data

    __call__: Callable[..., Any] = _call_impl


class TransformsCompose(torch.nn.Module):
    def __init__(self, transform: List[Any] or torch.nn.ModuleList[torch.nn.Module]):
        """
        Composes multiple transforms methods.
        :param transform: List of transforms to execute
        """
        assert transform, 'Please provide an non-empty list of torch modules (transforms)'
        super(TransformsCompose, self).__init__()
        self.transforms = transform

    def __call__(self, data: torch.Tensor or Dict[str, torch.Tensor],
                 *args, **kwargs) -> torch.Tensor or Dict[str, torch.Tensor]:
        """
        Iterative execution of transform methods
        :param data: data for transform execution
        :param args:
        :param kwargs:
        :return:
        """
        for transform in self.transforms:
            data = transform(data, *args, **kwargs)

        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
