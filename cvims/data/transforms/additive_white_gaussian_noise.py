#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Organization  : Technische Hochschule Ingolstadt
# Institution   : Computer Vision for Intelligent Mobility Systems (CVIMS)
# Created By    : Dominik Rößle
# =============================================================================
"""
Implementation for additive white gaussian noise
"""
# =============================================================================
# Imports
# =============================================================================
import torch
from cvims.data.transforms.utils import SingleInstanceTransforms, MultiInstanceTransforms, MultiModalTransforms, \
    MultiModalMultiInstanceTransforms
from typing import List


__all__ = [
    'TransformAdditiveWhiteGaussianNoise', 'TransformMultiInstanceAdditiveWhiteGaussianNoise',
    'TransformMultiModalAdditiveWhiteGaussianNoise', 'TransformMultiModalMultiInstanceAdditiveWhiteGaussianNoise',
    'apply_additive_white_gaussian_noise', 'apply_random_additive_white_gaussian_noise',
    'AdditiveWhiteGaussianNoiseImpl'
]


def apply_additive_white_gaussian_noise(
        tensor: torch.Tensor, snr: float, clip: bool, clip_minimum: float = -1.0, clip_maximum: float = 1.0,
        inplace: bool = False) -> torch.Tensor:
    """
    Applies additive white gaussian noise to the passed tensor. If tensor is a batch of tensors then the gaussian noise
    is calculated over all samples and not individually
    :param tensor: torch.Tensor where additive white gaussian noise will be applied
    :param snr: Signal-to-noise ratio
    :param clip: Boolean flag for clipping the values
    :param clip_minimum: Minimum clip value for noisy tensor. Only active if clip is True.
    :param clip_maximum: Maximum clip value for noisy tensor. Only active if clip is True.
    :param inplace: Perform on tensor if True otherwise the tensor gets cloned
    :return:
    """

    if not inplace:
        tensor = torch.clone(tensor)

    # flatten tensor
    init_size = tensor.size()
    flat_size = tensor.view(-1).size()

    rms = torch.sqrt(torch.mean(tensor ** 2))

    numerator = torch.float_power(rms, exponent=2)
    denominator = torch.as_tensor(10 ** (snr / 10), dtype=torch.float32)
    rms_noise = torch.sqrt(numerator / denominator)

    normal_mean = torch.zeros(flat_size)
    normal_std = rms_noise.expand(flat_size).T
    noise = torch.normal(mean=normal_mean, std=normal_std)
    tensor.add_(noise.view(*init_size))

    if clip:
        # clip the tensor
        tensor.clip_(min=clip_minimum, max=clip_maximum)

    return tensor


def apply_random_additive_white_gaussian_noise(
        tensor: torch.Tensor, snr_mean: float, snr_std: float, snr_max_val: float, snr_min_val: float,
        clip: bool, clip_minimum: float = -1.0, clip_maximum: float = 1.0, inplace: bool = False) -> torch.Tensor:
    """
    Applies additive white gaussian noise to the passed tensor. If tensor is a batch of tensors then the gaussian noise
    is calculated over all samples and not individually
    :param tensor: torch.Tensor where additive white gaussian noise will be applied
    :param snr_mean: Mean of signal-to-noise ratio
    :param snr_std: Standard deviation of signal-to-noise ratio
    :param snr_max_val: Maximum value of signal-to-noise ratio
    :param snr_min_val: Minimum value of signal-to-noise ratio
    :param clip: Boolean flag for clipping the values
    :param clip_minimum: Minimum clip value for noisy tensor. Only active if clip is True.
    :param clip_maximum: Maximum clip value for noisy tensor. Only active if clip is True.
    :param inplace: Perform on tensor if True otherwise the tensor gets cloned
    :return:
    """

    r_snr = torch.normal(mean=snr_mean, std=snr_std, size=(1,))
    snr = min(snr_max_val, max(snr_min_val, r_snr[0]))

    return apply_additive_white_gaussian_noise(
        tensor=tensor, snr=snr, clip=clip, clip_minimum=clip_minimum, clip_maximum=clip_maximum, inplace=inplace
    )


class AdditiveWhiteGaussianNoiseImpl:
    def __init__(self, snr_mean: float, snr_std: float, snr_max_val: float, snr_min_val: float,
                 clip: bool, clip_minimum: float = -1.0, clip_maximum: float = 1.0, inplace: bool = False) -> None:
        """
        Transforms module for additive white gaussian noise
        :param snr_mean: Mean of signal-to-noise ratio
        :param snr_std: Standard deviation of signal-to-noise ratio
        :param snr_max_val: Maximum value of signal-to-noise ratio
        :param snr_min_val: Minimum value of signal-to-noise ratio
        :param clip: Boolean flag for clipping the values
        :param clip_minimum: Minimum clip value for noisy tensor. Only active if clip is True.
        :param clip_maximum: Maximum clip value for noisy tensor. Only active if clip is True.
        :param inplace: Perform on tensor if True otherwise the tensor gets cloned
        """
        if clip:
            assert clip_minimum and clip_maximum

        self.snr_mean = snr_mean
        self.snr_std = snr_std
        self.snr_max_val = snr_max_val
        self.snr_min_val = snr_min_val
        self.clip = clip
        self.clip_minimum = clip_minimum
        self.clip_maximum = clip_maximum
        self.inplace = inplace

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = apply_random_additive_white_gaussian_noise(
            tensor=data, snr_mean=self.snr_mean, snr_std=self.snr_std,
            snr_min_val=self.snr_min_val, snr_max_val=self.snr_max_val,
            clip=self.clip, clip_minimum=self.clip_minimum, clip_maximum=self.clip_maximum,
            inplace=self.inplace
        )

        return data


class TransformAdditiveWhiteGaussianNoise(SingleInstanceTransforms, AdditiveWhiteGaussianNoiseImpl):
    def __init__(self, probability: float, snr_mean: float, snr_std: float, snr_max_val: float, snr_min_val: float,
                 clip: bool, clip_minimum: float = -1.0, clip_maximum: float = 1.0, inplace: bool = False) -> None:
        """
        Transforms module for additive white gaussian noise
        :param probability: Probability in range [0, 1]
        :param snr_mean: Mean of signal-to-noise ratio
        :param snr_std: Standard deviation of signal-to-noise ratio
        :param snr_max_val: Maximum value of signal-to-noise ratio
        :param snr_min_val: Minimum value of signal-to-noise ratio
        :param clip: Boolean flag for clipping the values
        :param clip_minimum: Minimum clip value for noisy tensor. Only active if clip is True.
        :param clip_maximum: Maximum clip value for noisy tensor. Only active if clip is True.
        :param inplace: Perform on tensor if True otherwise the tensor gets cloned
        """
        SingleInstanceTransforms.__init__(
            self=self, probability=probability, inplace=inplace
        )

        AdditiveWhiteGaussianNoiseImpl.__init__(
            self=self, snr_mean=snr_mean, snr_std=snr_std,
            snr_max_val=snr_max_val, snr_min_val=snr_min_val,
            clip=clip, clip_minimum=clip_minimum, clip_maximum=clip_maximum, inplace=inplace
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return AdditiveWhiteGaussianNoiseImpl.forward(self=self, data=data)


class TransformMultiInstanceAdditiveWhiteGaussianNoise(MultiInstanceTransforms, AdditiveWhiteGaussianNoiseImpl):
    def __init__(
            self, instance_probability: float,
            snr_mean: float, snr_std: float, snr_max_val: float, snr_min_val: float,
            clip: bool, clip_minimum: float = -1.0, clip_maximum: float = 1.0,
            individual_instance_probabilities: bool = True, inplace: bool = False,
            apply_instances_as_batch: bool = False) -> None:
        """
        Transforms module for additive white gaussian noise for multi-instance processing.
        :param instance_probability: Probability of applying transform method
        :param individual_instance_probabilities: Individual probabilities for instances (True / False). If False then
        either all instances or no instance is polluted
        :param apply_instances_as_batch: Applies additive white gaussian noise as one tensor if True and each instance
        separately if False
        :param snr_mean: Mean of signal-to-noise ratio
        :param snr_std: Standard deviation of signal-to-noise ratio
        :param snr_max_val: Maximum value of signal-to-noise ratio
        :param snr_min_val: Minimum value of signal-to-noise ratio
        :param clip: Boolean flag for clipping the values
        :param clip_minimum: Minimum clip value for noisy tensor. Only active if clip is True.
        :param clip_maximum: Maximum clip value for noisy tensor. Only active if clip is True.
        :param inplace: Perform on tensor if True otherwise the tensor gets cloned
        """
        MultiInstanceTransforms.__init__(
            self=self, individual_instance_probabilities=individual_instance_probabilities,
            instance_probability=instance_probability, apply_instances_as_batch=apply_instances_as_batch,
            inplace=inplace
        )

        AdditiveWhiteGaussianNoiseImpl.__init__(
            self=self, snr_mean=snr_mean, snr_std=snr_std,
            snr_max_val=snr_max_val, snr_min_val=snr_min_val,
            clip=clip, clip_minimum=clip_minimum, clip_maximum=clip_maximum, inplace=inplace
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return AdditiveWhiteGaussianNoiseImpl.forward(self=self, data=data)


class TransformMultiModalAdditiveWhiteGaussianNoise(MultiModalTransforms, AdditiveWhiteGaussianNoiseImpl):
    def __init__(
            self, modality_probability: float,
            snr_mean: float, snr_std: float, snr_max_val: float, snr_min_val: float,
            clip: bool, clip_minimum: float = -1.0, clip_maximum: float = 1.0,
            individual_modality_probabilities: bool = True, inplace: bool = False,
            restricted_modalities: str or List[str] = None) -> None:
        """
        Transforms module for additive white gaussian noise for multi-instance processing.
        :param modality_probability: Probability of applying transform method
        :param individual_modality_probabilities: Individual probabilities for instances (True / False). If False then
        either all instances or no instance is polluted
        :param restricted_modalities: String value or List of strings with the names of the modalities which should not
        be transformed. None or empty string results in all modalities accepted otherwise the passed names will
         be restricted
        :param snr_mean: Mean of signal-to-noise ratio
        :param snr_std: Standard deviation of signal-to-noise ratio
        :param snr_max_val: Maximum value of signal-to-noise ratio
        :param snr_min_val: Minimum value of signal-to-noise ratio
        :param clip: Boolean flag for clipping the values
        :param clip_minimum: Minimum clip value for noisy tensor. Only active if clip is True.
        :param clip_maximum: Maximum clip value for noisy tensor. Only active if clip is True.
        :param inplace: Perform on tensor if True otherwise the tensor gets cloned
        """
        MultiModalTransforms.__init__(
            self=self, individual_modality_probabilities=individual_modality_probabilities,
            modality_probability=modality_probability, restricted_modalities=restricted_modalities, inplace=inplace
        )

        AdditiveWhiteGaussianNoiseImpl.__init__(
            self=self, snr_mean=snr_mean, snr_std=snr_std,
            snr_max_val=snr_max_val, snr_min_val=snr_min_val,
            clip=clip, clip_minimum=clip_minimum, clip_maximum=clip_maximum, inplace=inplace
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return AdditiveWhiteGaussianNoiseImpl.forward(self=self, data=data)


class TransformMultiModalMultiInstanceAdditiveWhiteGaussianNoise(
    MultiModalMultiInstanceTransforms, AdditiveWhiteGaussianNoiseImpl
):
    def __init__(
            self, modality_probability: float, instance_probability: float,
            snr_mean: float, snr_std: float, snr_max_val: float, snr_min_val: float,
            clip: bool, clip_minimum: float = -1.0, clip_maximum: float = 1.0,
            individual_modality_probabilities: bool = True, individual_instance_probabilities: bool = True,
            inplace: bool = False, restricted_modalities: str or List[str] = None,
            apply_instances_as_batch: bool = False) -> None:
        """
        Transforms module for additive white gaussian noise for multi-instance processing.
        :param individual_modality_probabilities: Use individual probabilities for modalities if True. Use either
        all or no modalities if False. Both cases are based on the modality probability.
        :param individual_instance_probabilities: Use individual probabilities for instances if True. Use either
        all or no instances if False. Both cases are based on the instance probability.
        :param modality_probability: Probability of applying transform method to modalities.
        :param instance_probability: Probability of applying transform method to instances.
        :param restricted_modalities: Restrict certain modalities to get transformed.
        :param snr_mean: Mean of signal-to-noise ratio
        :param snr_std: Standard deviation of signal-to-noise ratio
        :param snr_max_val: Maximum value of signal-to-noise ratio
        :param snr_min_val: Minimum value of signal-to-noise ratio
        :param clip: Boolean flag for clipping the values
        :param clip_minimum: Minimum clip value for noisy tensor. Only active if clip is True.
        :param clip_maximum: Maximum clip value for noisy tensor. Only active if clip is True.
        :param inplace: Perform on tensor if True otherwise the tensor gets cloned
        :param apply_instances_as_batch: Applies transforms as one tensor if True or each instance separately if False
        """
        MultiModalMultiInstanceTransforms.__init__(
            self=self, individual_modality_probabilities=individual_modality_probabilities,
            individual_instance_probabilities=individual_instance_probabilities,
            modality_probability=modality_probability, instance_probability=instance_probability,
            apply_instances_as_batch=apply_instances_as_batch, restricted_modalities=restricted_modalities,
            inplace=inplace)

        AdditiveWhiteGaussianNoiseImpl.__init__(
            self=self, snr_mean=snr_mean, snr_std=snr_std,
            snr_max_val=snr_max_val, snr_min_val=snr_min_val,
            clip=clip, clip_minimum=clip_minimum, clip_maximum=clip_maximum, inplace=inplace
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return AdditiveWhiteGaussianNoiseImpl.forward(self=self, data=data)
