#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Organization  : Technische Hochschule Ingolstadt
# Institution   : Computer Vision for Intelligent Mobility Systems (CVIMS)
# Created By    : Dominik Rößle
# =============================================================================
"""
Implementation for random noise
"""
# =============================================================================
# Imports
# =============================================================================
import torch
from cvims.data.transforms.utils import SingleInstanceTransforms, MultiInstanceTransforms, MultiModalTransforms, \
    MultiModalMultiInstanceTransforms, normalize
from typing import List


__all__ = [
    'TransformRandomValueDistortion', 'TransformMultiInstanceRandomValueDistortion',
    'TransformMultiModalRandomValueDistortion', 'TransformMultiModalMultiInstanceRandomValueDistortion',
    'apply_random_noise_values', 'RandomValueDistortionImpl'
]


def apply_random_noise_values(tensor: torch.Tensor, random_percentage: float, inplace: bool = False) -> torch.Tensor:
    """
    Applies random noise to the input tensor based on a percentage value. The random value ranges between min and max
    value of the input tensor.
    :param tensor: torch.Tensor
    :param random_percentage: between 0 and 1
    :param inplace: Perform on tensor if True otherwise the tensor gets cloned
    :return:
    """
    assert 0.0 < random_percentage <= 1.0, 'Please provide a valid percentage value for applying random noise.'

    if not inplace:
        tensor = torch.clone(tensor)

    # modifiable tensor - we use this tensor for our random operations to keep operations for the passed tensor in-place
    mod_tensor = torch.ones(size=tensor.view(-1).size())

    # get ranges
    min_val, max_val = tensor.min(), tensor.max()

    # get amount of values to change
    val_count = mod_tensor.size()[0]
    # get random indices based on rand val percentage
    r_val_count = torch.round(torch.tensor([val_count * random_percentage])).type(torch.IntTensor)
    rand_idx = torch.randperm(n=val_count)[:r_val_count]
    # between 0 and 1
    rand_values = torch.rand(size=rand_idx.size())

    _range = max_val - min_val

    # get accurate borders - how much do the random values use their limits [0, 1]
    open_val_range_min = rand_values.min()
    open_val_range_max = 1 - rand_values.max()

    # scale between min and max with new borders
    new_rand_val_minimum = min_val + _range * open_val_range_min
    new_rand_val_maximum = max_val - _range * open_val_range_max

    rand_values, _, _ = normalize(
        tensor=rand_values, minimum=new_rand_val_minimum, maximum=new_rand_val_maximum
    )

    # assign random values to modifiable tensor
    mod_tensor[rand_idx] = 0.0
    # get rid of old values of random position
    tensor.mul_(mod_tensor.view(*tensor.size()))
    # set all mod_tensor vals to zero for add mul
    mod_tensor[:] = 0.0
    # apply random values to random positions
    mod_tensor[rand_idx] = rand_values
    # add new random value at random positions
    tensor.add_(mod_tensor.view(*tensor.size()))

    return tensor


class RandomValueDistortionImpl:
    def __init__(self, random_percentage: float, inplace: bool = False) -> None:
        assert 0.0 <= random_percentage <= 1.0
        self.rand_value_percentage = random_percentage
        self.inplace = inplace

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = apply_random_noise_values(
            tensor=data, random_percentage=self.rand_value_percentage, inplace=self.inplace
        )

        return data


class TransformRandomValueDistortion(SingleInstanceTransforms, RandomValueDistortionImpl):
    def __init__(self, probability: float, random_percentage: float, inplace: bool = False) -> None:
        """
        Transforms module for random noise.
        :param probability: Probability in range [0, 1]
        :param random_percentage: Percentage of random value distortion. Between 0 and 1.
        :param inplace: Perform on tensor if True otherwise the tensor gets cloned
        """
        SingleInstanceTransforms.__init__(
            self=self, probability=probability, inplace=inplace
        )

        RandomValueDistortionImpl.__init__(
            self=self, random_percentage=random_percentage
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return RandomValueDistortionImpl.forward(self=self, data=data)


class TransformMultiInstanceRandomValueDistortion(MultiInstanceTransforms, RandomValueDistortionImpl):
    def __init__(self, instance_probability: float, random_percentage: float, inplace: bool = False,
                 individual_instance_probabilities: bool = True, apply_instances_as_batch: bool = False) -> None:
        """
        Transforms module for random value distortion for multi-instance processing.
        :param instance_probability: Probability of applying transform method
        :param individual_instance_probabilities: Individual probabilities for instances (True / False). If False then
        either all instances or no instance is polluted
        :param apply_instances_as_batch: Applies additive white gaussian noise as one tensor if True and each instance
        separately if False
        :param random_percentage: Percentage of random value distortion. Between 0 and 1.
        :param inplace: Perform on tensor if True otherwise the tensor gets cloned
        """
        MultiInstanceTransforms.__init__(
            self=self, individual_instance_probabilities=individual_instance_probabilities,
            instance_probability=instance_probability, apply_instances_as_batch=apply_instances_as_batch,
            inplace=inplace
        )

        RandomValueDistortionImpl.__init__(
            self=self, random_percentage=random_percentage
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return RandomValueDistortionImpl.forward(self=self, data=data)


class TransformMultiModalRandomValueDistortion(MultiModalTransforms, RandomValueDistortionImpl):
    def __init__(self, modality_probability: float, random_percentage: float, inplace: bool = False,
                 individual_modality_probabilities: bool = True,
                 restricted_modalities: str or List[str] = None) -> None:
        """
        Transforms module for random value distortion for multi-modal processing.
        :param modality_probability: Probability of applying transform method
        :param individual_modality_probabilities: Individual probabilities for instances (True / False). If False then
        either all instances or no instance is polluted
        :param restricted_modalities: String value or List of strings with the names of the modalities which should not
        be transformed. None or empty string results in all modalities accepted otherwise the passed names will
         be restricted
        :param random_percentage: Percentage of random value distortion. Between 0 and 1.
        :param inplace: Perform on tensor if True otherwise the tensor gets cloned
        """
        MultiModalTransforms.__init__(
            self=self, individual_modality_probabilities=individual_modality_probabilities,
            modality_probability=modality_probability,
            restricted_modalities=restricted_modalities, inplace=inplace
        )

        RandomValueDistortionImpl.__init__(
            self=self, random_percentage=random_percentage
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return RandomValueDistortionImpl.forward(self=self, data=data)


class TransformMultiModalMultiInstanceRandomValueDistortion(
    MultiModalMultiInstanceTransforms, RandomValueDistortionImpl
):
    def __init__(self, modality_probability: float, instance_probability: float, random_percentage: float,
                 individual_modality_probabilities: bool = True, individual_instance_probabilities: bool = True,
                 inplace: bool = False, restricted_modalities: str or List[str] = None,
                 apply_instances_as_batch: bool = False) -> None:
        """
        Transforms module for random value distortion noise for multi-modal and multi-instance processing.
        :param individual_modality_probabilities: Use individual probabilities for modalities if True. Use either
        all or no modalities if False. Both cases are based on the modality probability.
        :param individual_instance_probabilities: Use individual probabilities for instances if True. Use either
        all or no instances if False. Both cases are based on the instance probability.
        :param modality_probability: Probability of applying transform method to modalities.
        :param instance_probability: Probability of applying transform method to instances.
        :param restricted_modalities: Restrict certain modalities to get transformed.
        :param inplace: Perform on tensor if True otherwise the tensor gets cloned
        :param apply_instances_as_batch: Applies transforms as one tensor if True or each instance separately if False
        :param random_percentage: Percentage of random value distortion. Between 0 and 1.
        """
        MultiModalMultiInstanceTransforms.__init__(
            self=self, individual_modality_probabilities=individual_modality_probabilities,
            individual_instance_probabilities=individual_instance_probabilities,
            modality_probability=modality_probability, instance_probability=instance_probability,
            apply_instances_as_batch=apply_instances_as_batch, restricted_modalities=restricted_modalities,
            inplace=inplace)

        RandomValueDistortionImpl.__init__(
            self=self, random_percentage=random_percentage
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return RandomValueDistortionImpl.forward(self=self, data=data)
