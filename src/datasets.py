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
import os
import copy
import math
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import logging
import json
from typing import List, Callable, Dict, Tuple
from abc import ABC, abstractmethod


class DigitDataset(Dataset):
    def __init__(self,
                 json_path: str,
                 read_file_function: Callable):
        """
        Instantiation for a digit dataset.
        :param json_path: Path to the dataset annotation file.
        :param read_file_function: Function callable to load the file from path.
        """
        assert isinstance(json_path, str), 'Please provide the path to the json annotation file.'
        assert os.path.isfile(json_path), 'The provided json path is not available.'

        self.json_path = json_path
        self.read_file_function = read_file_function

        annotations = DigitDataset.load_json_dataset(path=json_path)
        self._digit_annotations, self._labels, self.data_label_indices = DigitDataset.create_digit_labels_list(
            annotations=annotations
        )

    @property
    def digit_annotations(self):
        return self._digit_annotations

    @property
    def labels(self):
        return self._labels

    def unique_labels(self):
        return list(set(self._labels))

    @staticmethod
    def create_digit_labels_list(annotations: Dict[int, any]) -> Tuple[List, List, Dict]:
        # we make a flat lst of digits and values
        digits = []
        labels = []
        # separate digit idx
        label_data_idx = {}

        c = 0
        for digit in annotations:
            latest = c
            if not str(digit).isnumeric():
                continue

            for annotation in annotations[digit]:
                digits.append(annotation)
                labels.append(int(digit))
                c += 1

            label_data_idx[int(digit)] = [i for i in range(latest, c)]

        logging.info(msg='Completed loading digit annotation - label lists.')

        return digits, labels, label_data_idx

    @staticmethod
    def load_json_dataset(path: str):
        """
        Loads the json file and returns its content.
        :param path: Path to json file.
        :return: Dictionary of key digit and values of type list which contains the paths.
        """
        logging.info('Loading json annotation files.')

        assert path, 'Please provide a valid path.'
        assert os.path.isfile(path), 'The provided json path is not available.'

        # open annotation file
        with open(path, 'r') as f:
            annotation = json.load(f)
            if 'name' in annotation:
                del annotation['name']

            logging.info('Completed loading json annotations files.')
            return annotation

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        digit_annotation = self._digit_annotations[item]
        label = self._labels[item]

        # load images (extract images from data path)
        digit = self.read_file_function(digit_annotation['full_file_path'])

        label = torch.tensor(label, dtype=torch.long)

        return digit, label


class DynamicDatasetInterface(ABC, Dataset):
    def __init__(self, data_names, bags_list, labels_list,
                 transforms: torch.nn.Module):
        self.data_names = data_names
        self.bags_list = bags_list
        self.labels_list = labels_list
        self.transform = transforms

    @abstractmethod
    def load_data_bag(self, dynamic_bag, label):
        raise NotImplementedError()

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, item):
        dynamic_bag = self.bags_list[item]
        label = self.labels_list[item]

        # load data
        resolved_dyn_bag, label = self.load_data_bag(dynamic_bag=dynamic_bag, label=label)

        if self.transform:
            resolved_dyn_bag = self.transform(
                resolved_dyn_bag
            )

        return resolved_dyn_bag, label


class MultiInstanceDataset(DynamicDatasetInterface):
    def __init__(self,
                 name: str,
                 json_path: str,
                 read_file_function: Callable,
                 mean_instances_per_bag: int = 2,
                 std_instances_per_bag: float = 0.0,
                 transforms: torch.nn.Module = None,
                 return_as_dict: bool = False):
        """
        Instantiation for a digit dataset
        :param name: Name of the dataset
        :param json_path: Path to the dataset annotation file
        :param read_file_function: Function callable to load the file from path
        :param mean_instances_per_bag: Mean instance for every instance of the bag. An instance is considered
        the content of the inner bags
        :param std_instances_per_bag: Variance of the inner bags
        :param transforms:
        :param return_as_dict: Returns the item in dict format { name: tensor }
        """
        assert isinstance(json_path, str), 'Please provide the path to the json annotation file.'
        assert os.path.isfile(json_path), 'The provided json path is not available.'

        self.name = name
        self.json_path = json_path
        self.read_file_function = read_file_function
        self.mean_instances_per_bag = mean_instances_per_bag
        self.std_instances_per_bag = std_instances_per_bag
        self.return_as_dict = return_as_dict

        annotations = DigitDataset.load_json_dataset(path=json_path)
        digit_annotations, labels = self.create_multi_digit_labels_list(annotations=annotations)

        super().__init__(
            data_names=[name], bags_list=digit_annotations, labels_list=labels, transforms=transforms
        )

    def create_multi_digit_labels_list(self, annotations: Dict[int, any]) -> Tuple[List, List]:
        c_annotations = copy.deepcopy(annotations)

        multi_instance_list = []
        labels = []

        for digit in c_annotations:
            if not str(digit).isnumeric():
                continue

            while len(c_annotations[digit]) >= self.mean_instances_per_bag:
                num_instances = torch.normal(self.mean_instances_per_bag, self.std_instances_per_bag, (1,))
                num_instances = torch.round(num_instances).type(torch.IntTensor)
                num_instances = max(1, num_instances)
                # num_instances = min(len(c_annotations[digit]), num_instances)

                # random indices
                indices = torch.randperm(len(c_annotations[digit]))[:num_instances].tolist()

                combined_annotations = []
                for index in sorted(indices, reverse=True):
                    combined_annotations.append(c_annotations[digit][index])
                    del c_annotations[digit][index]

                multi_instance_list.append(combined_annotations)
                labels.append(int(digit))

        logging.info(msg='Completed loading digit annotation - label lists.')

        return multi_instance_list, labels

    def load_data_bag(self, dynamic_bag, label):
        instances = []
        # load data
        for i, annotation in enumerate(dynamic_bag):
            data = self.read_file_function(annotation['full_file_path'])
            instances.append(data)

        instances = torch.stack(instances, dim=0)

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)

        if self.return_as_dict:
            return {self.name: instances}, label
        else:
            return instances, label


class DynamicInstanceDataset(DynamicDatasetInterface):
    def __init__(self,
                 batch_size: int,
                 full_json_path_list: List[str],
                 data_path_names: List[str],
                 read_file_functions: List[Callable],
                 num_dynamic_bags: int,
                 mean_instances_per_bag: int = 2,
                 std_instances_per_bag: float = 2,
                 transforms: torch.nn.Module = None):
        """
        Instantiation for the dynamic dataset. The dataset returns items of shape (b, i, ...) where b is the batch size
        and i is the amount of instances. The instances across the modalities can vary according to the mean and std
        parameters. The minimum amount of instances of a modality is 1 no matter the normal distribution parameters.
        :param batch_size: Generates batches inside the dataset. You can only use a data loader with batch size 1 with
        this dataset. The dimensional level of the labels is also increased by batch size which results in a tensor of
        shape (b, label) for labels
        :param full_json_path_list: List of datasets to include. Please make sure that the json structure is ensured.
        :param data_path_names: List of the data names in the same order than the full_json_path_list provided.
        :param read_file_functions: Provide a function which has full_data_path as argument and is capable of loading
        the data from its path. Please provide the functions in the same order as the full json path list.
        :param num_dynamic_bags: Size of the dataset. The dynamic bag contains multiple inner bags. E. g. One bag
        contains an inner mnist bag and an inner svhn bag. The bags are created by random, which means that the
        num_dynamic_bags can contain data items among all bags multiple times.
        :param mean_instances_per_bag: Mean instance for every instance of the bag. An instance is considered
        the content of the inner bags.
        :param std_instances_per_bag: Variance of the inner bags
        :param transforms: Data transforms
        """
        assert batch_size >= 1
        self.batch_size = batch_size

        assert isinstance(full_json_path_list, list) and full_json_path_list, "Please provide a non-empty json " \
                                                                              "annotation path list."
        self.full_json_path_list = full_json_path_list
        if data_path_names:
            assert isinstance(data_path_names, list)
        self.data_path_names = data_path_names
        self.read_file_functions = {n: fn for n, fn in zip(data_path_names, read_file_functions)}
        # num of bags divided by batch size because batch size is part of the data return
        self.num_dynamic_bags = max(1, math.ceil(num_dynamic_bags // batch_size))
        self.mean_instances_per_bag = mean_instances_per_bag
        self.std_instances_per_bag = std_instances_per_bag

        json_info = DynamicDataset.load_json_dataset(path_list=full_json_path_list, path_name_list=data_path_names)
        bags_list, labels_list = self._create_dynamic_bags(json_info=json_info)
        super().__init__(
            data_names=data_path_names, bags_list=bags_list, labels_list=labels_list, transforms=transforms
        )

    def _create_dynamic_bags(self, json_info):
        """
        Prepares the data structure for the bags, inner bags and instances.
        :param json_info: result of load json dataset
        :return:
        """

        json_copy = copy.deepcopy(json_info)

        bags_list = []
        labels_list = []

        data_names = list(json_copy.keys())
        used_annotations = {k: {} for k in data_names}
        for k in used_annotations:
            for n in range(0, 10):
                used_annotations[k][n] = []

        logging.info('Preparing data bags.')

        # load all json files
        for i in range(self.num_dynamic_bags):
            # calculate bag length for every data_name
            bag_lengths = {}
            for data_name in data_names:
                bag_length = torch.normal(self.mean_instances_per_bag, self.std_instances_per_bag, (1,)).detach()
                bag_length = torch.round(bag_length).type(torch.IntTensor)
                bag_length = max(1, bag_length)

                bag_lengths[data_name] = bag_length

            # which number should be chosen for the dynamic bag
            rand_numbers = torch.randint(low=0, high=10, size=(self.batch_size,)).detach()

            # due to the stacking of items, the bag length (instances) must stay the same for one data output (across
            # one batch)
            batch_bags = {}
            for data_name in data_names:
                inner_bags = []
                # load the data from the json info for the rand_number
                for rand_number in rand_numbers:
                    r_number = rand_number.item()
                    json_data = json_copy[data_name]
                    # get the items of the random number
                    data_annotations = json_data[str(r_number)]
                    # mix already used into sampling
                    if len(data_annotations) <= 0:
                        # use only used ones
                        used_cache = used_annotations[data_name][r_number]
                        indices = torch.randperm(len(used_cache))[:bag_lengths[data_name]]
                        samples = [used_cache[index] for index in indices]
                    elif len(data_annotations) < bag_lengths[data_name]:
                        # add residuals
                        used_cache = used_annotations[data_name][r_number]
                        samples = data_annotations + used_cache
                        indices = list(range(0, len(data_annotations)))
                        # add randoms from used cache
                        # add the length of indices to avoid indices inside the residuals range
                        samples_adds = torch.randperm(len(samples) - len(indices)) + len(indices)
                        samples_adds = samples_adds[:bag_lengths[data_name] - len(indices)]
                        indices.extend(samples_adds.tolist())
                        samples = [samples[index] for index in indices]
                    else:
                        # use only new ones (data annotations)
                        indices = torch.randperm(len(data_annotations))[:bag_lengths[data_name]].tolist()

                        # used sorted and reversed to not shift the index inappropriate
                        samples = []
                        for index in sorted(indices, reverse=True):
                            annotation = data_annotations[index]
                            samples.append(annotation)
                            del data_annotations[index]
                        # add samples to used annotations
                        used_annotations[data_name][r_number].extend(samples)

                    inner_bags.append(samples)

                # add them to dict
                batch_bags[data_name] = inner_bags

            bags_list.append(batch_bags)
            labels_list.append(rand_numbers)

        logging.info('Completed preparing data bags process.')

        return bags_list, labels_list

    def load_data_bag(self, dynamic_bag: Dict, label):
        # load data
        resolved_batches = {}
        for data_name in dynamic_bag:
            resolved_dyn_bag = []
            batch_annotations = dynamic_bag[data_name]
            for batch_annotation in batch_annotations:
                annotations_list = []
                for annotation in batch_annotation:
                    read_file_fn = self.read_file_functions[data_name]
                    file = read_file_fn(annotation['full_file_path'])
                    annotations_list.append(file)
                resolved_dyn_bag.append(torch.unsqueeze(torch.stack(annotations_list), dim=0))
            resolved_batches[data_name] = torch.cat(resolved_dyn_bag, dim=0)

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)

        return resolved_batches, label


class DynamicDataset(DynamicDatasetInterface):
    def __init__(self,
                 full_json_path_list: List[str],
                 data_path_names: List[str],
                 read_file_functions: List[Callable],
                 num_dynamic_bags: int,
                 mean_instances_per_bag: int = 2,
                 std_instances_per_bag: float = 2,
                 transforms: torch.nn.Module = None):
        """
        Instantiation for the dynamic dataset.
        :param full_json_path_list: List of datasets to include. Please make sure that the json structure is ensured.
        :param data_path_names: List of the data names in the same order than the full_json_path_list provided.
        :param read_file_functions: Provide a function which has full_data_path as argument and is capable of loading
        the data from its path. Please provide the functions in the same order as the full json path list.
        :param num_dynamic_bags: Size of the dataset. The dynamic bag contains multiple inner bags. E. g. One bag
        contains an inner mnist bag and an inner svhn bag. The bags are created by random, which means that the
        num_dynamic_bags can contain data items among all bags multiple times.
        :param mean_instances_per_bag: Mean instance for every instance of the bag. An instance is considered
        the content of the inner bags.
        :param std_instances_per_bag: Variance of the inner bags
        :param transforms: Data transforms
        """
        assert isinstance(full_json_path_list, list) and full_json_path_list, "Please provide a non-empty json " \
                                                                              "annotation path list."
        self.full_json_path_list = full_json_path_list
        if data_path_names:
            assert isinstance(data_path_names, list)
        self.read_file_functions = {n: fn for n, fn in zip(data_path_names, read_file_functions)}
        self.num_dynamic_bags = num_dynamic_bags
        self.mean_instances_per_bag = mean_instances_per_bag
        self.std_instances_per_bag = std_instances_per_bag

        json_info = DynamicDataset.load_json_dataset(path_list=full_json_path_list, path_name_list=data_path_names)
        bags_list, labels_list = self._create_dynamic_bags(json_info=json_info)
        DynamicDatasetInterface.__init__(
            self=self, data_names=data_path_names, bags_list=bags_list, labels_list=labels_list, transforms=transforms
        )

    @staticmethod
    def load_json_dataset(path_list: List[str], path_name_list: List[str] = None):
        """
        Loads the json file and returns its content.
        :param path_list: List of absolute paths to the json annotation files.
        :param path_name_list:
        :return: Dictionary of key "data_name" and values of type dict with key: 0-9 and value: data_info
        """
        logging.info('Loading json annotations files.')

        if path_name_list is not None:
            assert len(path_name_list) == len(path_list)

        data_dict = {}
        for i, path in enumerate(tqdm(path_list)):
            # open annotation file
            with open(path, 'r') as f:
                annotation = json.load(f)
            # read annotation file
            if path_name_list is None:
                data_name = annotation['name']
            else:
                data_name = path_name_list[i]
            data_dict[data_name] = annotation
            # remove the redundant information
            del data_dict[data_name]['name']

        logging.info('Completed loading json annotations files.')

        return data_dict

    def _create_dynamic_bags(self, json_info):
        """
        Prepares the data structure for the bags, inner bags and instances.
        :param json_info: result of load json dataset
        :return:
        """

        json_copy = copy.deepcopy(json_info)

        bags_list = []
        labels_list = []

        data_names = list(json_copy.keys())
        used_annotations = {k: {} for k in data_names}
        for k in used_annotations:
            for n in range(0, 10):
                used_annotations[k][n] = []

        logging.info('Preparing data bags.')

        # load all json files
        for i in range(self.num_dynamic_bags):
            # calculate bag length for every data_name
            bag_lengths = {}
            for data_name in data_names:
                bag_length = torch.normal(self.mean_instances_per_bag, self.std_instances_per_bag, (1,))
                bag_length = torch.round(bag_length).type(torch.IntTensor)
                if bag_length < 0:
                    bag_length = 0

                bag_lengths[data_name] = bag_length

            # check if at least one bag has more than zero entries
            if all(bag_lengths[bag_name] == 0 for bag_name in bag_lengths):
                # add bag_length == 1 to one random entry
                name_index = torch.randint(low=0, high=len(data_names), size=(1,)).detach().item()
                bag_lengths[data_names[name_index]] = 1

            # which number should be chosen for this particular dynamic bag
            rand_number = torch.randint(low=0, high=10, size=(1,)).detach().item()

            # load the data from the json info for the rand_number
            inner_bags = {}
            for data_name in data_names:
                json_data = json_copy[data_name]
                # get the items of the random number
                data_annotations = json_data[str(rand_number)]
                # mix already used into sampling
                if len(data_annotations) <= 0:
                    # use only used ones
                    used_cache = used_annotations[data_name][rand_number]
                    sample_indices = list(range(0, len(used_cache)))
                    indices = torch.randperm(len(used_cache))[:bag_lengths[data_name]]
                    samples = [used_cache[index] for index in indices]
                elif len(data_annotations) < bag_lengths[data_name]:
                    # add residuals
                    used_cache = used_annotations[data_name][rand_number]
                    samples = data_annotations + used_cache
                    indices = list(range(0, len(data_annotations)))
                    # add randoms from used cache
                    # add the length of indices to avoid indices inside the residuals range
                    samples_adds = torch.randperm(len(samples) - len(indices)) + len(indices)
                    samples_adds = samples_adds[:bag_lengths[data_name] - len(indices)]
                    indices.extend(samples_adds.tolist())
                    samples = [samples[index] for index in indices]
                else:
                    # use only new ones (data annotations)
                    indices = torch.randperm(len(data_annotations))[:bag_lengths[data_name]].tolist()

                    # used sorted and reversed to not shift the index inappropriate
                    samples = []
                    for index in sorted(indices, reverse=True):
                        annotation = data_annotations[index]
                        samples.append(annotation)
                        del data_annotations[index]
                    # add samples to used annotations
                    used_annotations[data_name][rand_number].extend(samples)

                inner_bags[data_name] = samples

            bags_list.append(inner_bags)
            labels_list.append(rand_number)

        logging.info('Completed preparing data bags process.')
        return bags_list, labels_list

    def load_data_bag(self, dynamic_bag: Dict, label):
        # load data
        resolved_dyn_bag = {}
        for data_name in dynamic_bag:
            resolved_dyn_bag[data_name] = []
            annotations = dynamic_bag[data_name]
            for annotation in annotations:
                read_file_fn = self.read_file_functions[data_name]
                file = read_file_fn(annotation['full_file_path'])
                resolved_dyn_bag[data_name].append(file)
            if resolved_dyn_bag[data_name]:
                resolved_dyn_bag[data_name] = torch.stack(resolved_dyn_bag[data_name])
            else:
                # do not write an empty dict entry!
                del resolved_dyn_bag[data_name]

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)

        return resolved_dyn_bag, label
