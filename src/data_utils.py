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
from cvims.data.loaders import create_data_loader
from src.datasets import DigitDataset, MultiInstanceDataset, DynamicDataset, DynamicInstanceDataset


def load_data_path_list_and_functions(perceiver_params, key_path, key_data_method):
    """
    Splits the perceiver information into keys, data paths and data load functions
    """
    paths = [perceiver_params[tp]['data_information'][key_path] for tp in perceiver_params]
    functions = [perceiver_params[tp]['data_information'][key_data_method]
                 for tp in perceiver_params]

    return list(perceiver_params.keys()), paths, functions


def build_train_val_digit_dataset(perceiver_params):
    """
    :param perceiver_params: Parsed config section (perceivers)
    """
    _, train_path_list, train_load_functions = load_data_path_list_and_functions(
        perceiver_params,
        'train_data_path',
        'train_load_data_method')

    _, val_path_list, val_load_functions = load_data_path_list_and_functions(
        perceiver_params,
        'val_data_path',
        'val_load_data_method')

    train_dataset = DigitDataset(
        json_path=train_path_list[0],
        read_file_function=train_load_functions[0]
    )

    val_dataset = DigitDataset(
        json_path=val_path_list[0],
        read_file_function=val_load_functions[0]
    )

    return train_dataset, val_dataset


def build_train_val_dynamic_instances_dataset(perceiver_params, data_loader_information, return_as_dict: bool = False):
    """
    :param perceiver_params: Parsed config section (perceivers)
    :param data_loader_information: Parser config section (data_settings['data_loader'])
    :param return_as_dict
    """
    name = list(perceiver_params.keys())[0]

    _, train_path_list, train_load_functions = load_data_path_list_and_functions(
        perceiver_params,
        'train_data_path',
        'train_load_data_method')

    _, val_path_list, val_load_functions = load_data_path_list_and_functions(
        perceiver_params,
        'val_data_path',
        'val_load_data_method')

    # data information

    # train data information
    train_data_info = data_loader_information['data']['train']
    t_mean_instances_per_bag = train_data_info['mean_instances_per_bag']
    t_std_instances_per_bag = train_data_info['std_instances_per_bag']

    # val data information
    val_data_info = data_loader_information['data']['val']
    v_mean_instances_per_bag = val_data_info['mean_instances_per_bag']
    v_std_instances_per_bag = val_data_info['std_instances_per_bag']

    train_dataset = MultiInstanceDataset(
        name=name,
        json_path=train_path_list[0],
        read_file_function=train_load_functions[0],
        mean_instances_per_bag=t_mean_instances_per_bag,
        std_instances_per_bag=t_std_instances_per_bag,
        return_as_dict=return_as_dict
    )

    val_dataset = MultiInstanceDataset(
        name=name,
        json_path=val_path_list[0],
        read_file_function=val_load_functions[0],
        mean_instances_per_bag=v_mean_instances_per_bag,
        std_instances_per_bag=v_std_instances_per_bag,
        return_as_dict=return_as_dict
    )

    return train_dataset, val_dataset


def build_train_val_dynamic_bag_dataset(perceiver_params, data_loader_information):
    """
    :param perceiver_params: Parsed config section (perceivers)
    :param data_loader_information: Parser config section (data_settings['data_loader'])
    """
    train_keys, train_path_list, train_load_functions = load_data_path_list_and_functions(
        perceiver_params,
        'train_data_path',
        'train_load_data_method')

    val_keys, val_path_list, val_load_functions = load_data_path_list_and_functions(
        perceiver_params,
        'val_data_path',
        'val_load_data_method')

    # data information
    batch_size = data_loader_information['batch_size']
    load_type = data_loader_information['load_type']

    # train data information
    train_data_info = data_loader_information['data']['train']
    t_num_bags = train_data_info['num_bags']
    t_mean_instances_per_bag = train_data_info['mean_instances_per_bag']
    t_std_instances_per_bag = train_data_info['std_instances_per_bag']

    # val data information
    val_data_info = data_loader_information['data']['val']
    v_num_bags = val_data_info['num_bags']
    v_mean_instances_per_bag = val_data_info['mean_instances_per_bag']
    v_std_instances_per_bag = val_data_info['std_instances_per_bag']

    if load_type == 'static':
        train_dataset = DynamicDataset(
            full_json_path_list=train_path_list,
            data_path_names=train_keys,
            read_file_functions=train_load_functions,
            num_dynamic_bags=t_num_bags,
            mean_instances_per_bag=t_mean_instances_per_bag,
            std_instances_per_bag=t_std_instances_per_bag
        )

        val_dataset = DynamicDataset(
            full_json_path_list=val_path_list,
            data_path_names=val_keys,
            read_file_functions=val_load_functions,
            num_dynamic_bags=v_num_bags,
            mean_instances_per_bag=v_mean_instances_per_bag,
            std_instances_per_bag=v_std_instances_per_bag
        )
    elif load_type == 'dynamic':
        train_dataset = DynamicInstanceDataset(
            batch_size=batch_size,
            full_json_path_list=train_path_list,
            data_path_names=train_keys,
            read_file_functions=train_load_functions,
            num_dynamic_bags=t_num_bags,
            mean_instances_per_bag=t_mean_instances_per_bag,
            std_instances_per_bag=t_std_instances_per_bag
        )

        val_dataset = DynamicInstanceDataset(
            batch_size=batch_size,
            full_json_path_list=val_path_list,
            data_path_names=val_keys,
            read_file_functions=val_load_functions,
            num_dynamic_bags=v_num_bags,
            mean_instances_per_bag=v_mean_instances_per_bag,
            std_instances_per_bag=v_std_instances_per_bag
        )
    else:
        raise NotImplementedError(f'Cannot load dataset for load type with name: {load_type}')

    return train_dataset, val_dataset


def init_dynamic_bag_data_loader(dataset_paths, dataset_path_names, data_load_functions,
                                 num_bags, mean_instance_per_bag, std_instance_per_bag,
                                 load_type, **data_loader_kwargs):
    """
    Initializes dataset and data loader for dynamic (modal and instance or only instance) datasets
    """
    assert len(dataset_paths) == len(data_load_functions)

    assert load_type == 'static' or load_type == 'dynamic'

    if load_type == 'static':
        ds = DynamicDataset(
            full_json_path_list=dataset_paths,
            data_path_names=dataset_path_names,
            read_file_functions=data_load_functions,
            num_dynamic_bags=num_bags,
            mean_instances_per_bag=mean_instance_per_bag,
            std_instances_per_bag=std_instance_per_bag
        )
    elif load_type == 'dynamic':
        batch_size = data_loader_kwargs['batch_size']
        ds = DynamicInstanceDataset(
            batch_size=batch_size,
            full_json_path_list=dataset_paths,
            data_path_names=dataset_path_names,
            read_file_functions=data_load_functions,
            num_dynamic_bags=num_bags,
            mean_instances_per_bag=mean_instance_per_bag,
            std_instances_per_bag=std_instance_per_bag
        )
    else:
        raise NotImplementedError()

    if 'workers' in data_loader_kwargs:
        workers = data_loader_kwargs['workers']
        del data_loader_kwargs['workers']
    else:
        workers = 1

    dl = create_data_loader(
        dataset=ds, workers=workers, **data_loader_kwargs
    )

    return ds, dl


def init_digit_datasets(dataset_path_names, dataset_paths, data_load_functions):
    """
    Initializes digit datasets (usual single instance dataset) for each modality
    """
    d_datasets = {}

    for name, path, fn in zip(dataset_path_names, dataset_paths, data_load_functions):
        digit_dataset = DigitDataset(
            json_path=path,
            read_file_function=fn
        )

        d_datasets[name] = digit_dataset

    return d_datasets
