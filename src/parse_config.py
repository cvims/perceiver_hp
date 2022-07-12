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
import argparse
import configparser
import copy
import logging
import torch
import torchaudio
import torchvision
from PIL import Image
from einops.layers.torch import Rearrange
from torch.nn import Sequential
from typing import Tuple, Dict
from cvims.data.transforms.utils import normalize


def load_configuration(model_run_path: str) -> Tuple[str, Dict]:
    model_path = os.path.join(model_run_path, 'model_info', 'best_model.pth')
    config_path = os.path.join(model_run_path, 'config.ini')

    parsed_arguments = parse_arguments(config_file_path=config_path)

    return model_path, parsed_arguments


def read_perceiver_config_files(config_file_path: str, config_section) -> Dict:
    """
    Read config path and load perceiver configs
    :param config_file_path: Path to current config file
    :param config_section: Config section PERCEIVERS
    :return:
    """
    conf_root_path = os.path.realpath(os.path.join(config_file_path, config_section['PERCEIVER_CONFIG_PATH']))
    perceiver_config_file_names = str(config_section['PERCEIVER_CONFIG_FILE_NAMES']).replace(' ', '').split(',')

    perceiver_configs = {}
    for perceiver_name in perceiver_config_file_names:
        config_parser = configparser.ConfigParser()
        config_parser.read(os.path.join(conf_root_path, perceiver_name + '.ini'))

        perceiver_configs[perceiver_name] = {}
        perceiver_configs[perceiver_name]['config'] = config_parser

    return perceiver_configs


def read_perceiver_configurations(perceivers) -> Dict:
    """
    Loads the perceiver configurations from the loaded configuration file
    :param perceivers:
    :return:
    """

    perceiver_dict = copy.deepcopy(perceivers)

    for perceiver_name in perceiver_dict:
        perceiver_config = perceiver_dict[perceiver_name]['config']
        if 'MODEL_PARAMETERS' not in perceiver_config:
            continue
        # pull sections and attributes
        model_parameters = perceiver_config['MODEL_PARAMETERS']
        perceiver_dict[perceiver_name]['model_parameters'] = parse_perceiver_model_parameters(
            model_parameters
        )

        # read data section
        _data = perceiver_config['DATA']
        data_sec = {
            'train_data_path': _data['TRAIN_DATA_PATH'],
            'val_data_path': _data['VAL_DATA_PATH'],
            '__test_data_path__': _data['__TEST_DATA_PATH__'],
            'train_load_data_method':  map_data_function(type_name=_data['TRAIN_LOAD_DATA_METHOD']),
            'val_load_data_method': map_data_function(type_name=_data['VAL_LOAD_DATA_METHOD']),
            '__test_load_data_method__': map_data_function(type_name=_data['__TEST_LOAD_DATA_METHOD__']),
        }

        perceiver_dict[perceiver_name]['data_information'] = data_sec

    return perceiver_dict


def parse_perceiver_model_parameters(parameters) -> Dict:
    parsed_parameters = {
        'num_latents': int(parameters['num_latents']),
        'latent_dim': int(parameters['latent_dim']),
        'input_channels': int(parameters['input_channels']),
        'input_axis': int(parameters['input_axis']),
        'self_per_cross_attention': int(parameters['self_per_cross_attention']),
        'cross_heads': int(parameters['cross_heads']),
        'cross_dim_head': int(parameters['cross_dim_head']),
        'cross_attention_dropout': float(parameters['cross_attention_dropout']),
        'latent_heads': int(parameters['latent_heads']),
        'latent_dim_head': int(parameters['latent_dim_head']),
        'latent_attention_dropout': float(parameters['latent_attention_dropout']),
        'tie_weight_pos_cross_attention': int(parameters['tie_weight_pos_cross_attention']),
        'tie_weight_pos_latent_attention': int(parameters['tie_weight_pos_latent_attention']),
        'iterative_count': int(parameters['iterative_count']),
        'fourier_encode_data': parameters.getboolean('fourier_encode_data'),
        'num_freq_bands': int(parameters['num_freq_bands']),
        'max_freq': float(parameters['max_freq']),
        'cross_attention_feed_forward_dropout': float(parameters['cross_attention_feed_forward_dropout']),
        'latent_attention_feed_forward_dropout': float(parameters['latent_attention_feed_forward_dropout'])
    }

    if 'use_hopfield_pooling' in parameters:
        parsed_parameters['use_hopfield_pooling'] = parameters.getboolean('use_hopfield_pooling')

        if parsed_parameters['use_hopfield_pooling']:
            parsed_parameters['hopfield_dim_head'] = int(parameters['hopfield_dim_head'])
            parsed_parameters['hopfield_heads'] = int(parameters['hopfield_heads'])
            parsed_parameters['hopfield_latent_attention_pos'] = int(parameters['hopfield_latent_attention_pos'])
            parsed_parameters['hopfield_max_update_steps'] = int(parameters['hopfield_max_update_steps'])
            parsed_parameters['hopfield_scaling'] = float(parameters['hopfield_scaling'])
            parsed_parameters['hopfield_dropout'] = float(parameters['hopfield_dropout'])

    return parsed_parameters


def parse_model_attributes(model_attributes) -> Dict:
    attributes = {}
    add_to_dict_if_exists(attributes, 'fusion_n', model_attributes, None, 'FUSION_N', model_attributes.getint)
    add_to_dict_if_exists(attributes, 'use_hopfield_pooling_fusion', model_attributes, None,
                          'USE_HOPFIELD_POOLING_FUSION', model_attributes.getboolean)

    if 'use_hopfield_pooling_fusion' in attributes and attributes['use_hopfield_pooling_fusion']:
        attributes['hopfield_fusion_dim_head'] = int(model_attributes['hopfield_fusion_dim_head'])
        attributes['hopfield_fusion_heads'] = int(model_attributes['hopfield_fusion_heads'])
        attributes['hopfield_fusion_max_update_steps'] = int(model_attributes['hopfield_fusion_max_update_steps'])
        attributes['hopfield_fusion_scaling'] = float(model_attributes['hopfield_fusion_scaling'])
        attributes['hopfield_fusion_dropout'] = float(model_attributes['hopfield_fusion_dropout'])

    # some not default parameters?
    # add them at the end
    unparsed = [key for key in list(model_attributes.keys()) if key not in attributes]
    attributes.update({k: model_attributes[k] for k in unparsed})

    return attributes


def add_to_dict_if_exists(dict_to_add, new_key, parsed_config, config_section, config_option, parse_fn):
    if config_section is None:
        if config_option in parsed_config:
            dict_to_add[new_key] = parse_fn(config_option)
    elif config_section is not None:
        if isinstance(parsed_config, configparser.ConfigParser):
            if parsed_config.has_option(config_section, config_option):
                dict_to_add[new_key] = parse_fn(config_section, config_option)
    else:
        logging.error(f'Could not add {new_key} to dictionary.')


def create_config_dict(config_file_path: str) -> Dict:
    # parse all config entries to to dict
    config_parser = configparser.ConfigParser()
    config_parser.read(config_file_path)

    parsed_dict = {
        'root_config': config_parser
    }

    # read perceiver config files
    perceiver_configs = read_perceiver_config_files(os.path.dirname(config_file_path), config_parser['PERCEIVERS'])

    # read perceicer configurations / parameters
    parsed_dict['perceivers'] = read_perceiver_configurations(perceiver_configs)

    if config_parser.has_section('MODEL_ATTRIBUTES'):
        parsed_dict['model_attributes'] = parse_model_attributes(config_parser['MODEL_ATTRIBUTES'])

    # read model parameters
    model_param_dict = {
        'learning_rate': config_parser.getfloat('MODEL_PARAMETERS', 'LEARNING_RATE'),
        'epochs': config_parser.getint('MODEL_PARAMETERS', 'EPOCHS'),
    }

    # add to dict
    parsed_dict['model_parameters'] = model_param_dict

    # general
    general_dict = {
        'log_dir': os.path.join(config_parser.get('GENERAL', 'LOG_DIR')),
        'use_tensorboard': config_parser.getboolean('GENERAL', 'USE_TENSORBOARD'),
        'save_best_model_only': config_parser.getboolean('GENERAL', 'SAVE_BEST_MODEL_ONLY'),
    }

    # add to dict
    parsed_dict['general'] = general_dict

    # early stopping parameters
    es_dict = {
        'optimization': None,
        'active': False
    }

    if config_parser.has_section('EARLY_STOPPING'):
        if config_parser.getboolean('EARLY_STOPPING', 'ACTIVE'):
            es_dict['active'] = True
            es_dict['patience'] = config_parser.getint('EARLY_STOPPING', 'PATIENCE')
            es_dict['delta'] = config_parser.getfloat('EARLY_STOPPING', 'DELTA')
            optimization = config_parser.get('EARLY_STOPPING', 'OPTIMIZATION')
            assert optimization.lower() == 'minimize' or optimization.lower() == 'maximize'
            es_dict['optimization'] = optimization

    # add to dict
    parsed_dict['early_stopping'] = es_dict

    # data loader settings
    data_dict = {'data_loader': {
        'batch_size': config_parser.getint('DATA_LOADER', 'BATCH_SIZE')
        }
    }
    data_dict['data_loader']['load_type'] = config_parser.get('DATA_LOADER', 'LOAD_TYPE')

    add_to_dict_if_exists(data_dict['data_loader'], 'generation_seed',
                          config_parser, 'DATA_LOADER', 'GENERATION_SEED', config_parser.getint)
    data_dict['data_loader']['data'] = {'train': {}, 'val': {}}
    train_sec = data_dict['data_loader']['data']['train']

    add_to_dict_if_exists(train_sec, 'num_bags',
                          config_parser, 'DATA_LOADER', 'TRAIN_NUM_BAGS', config_parser.getint)
    add_to_dict_if_exists(train_sec, 'mean_instances_per_bag',
                          config_parser, 'DATA_LOADER', 'TRAIN_MEAN_INSTANCES_PER_PAG', config_parser.getint)
    add_to_dict_if_exists(train_sec, 'std_instances_per_bag',
                          config_parser, 'DATA_LOADER', 'TRAIN_STD_INSTANCES_PER_BAG', config_parser.getfloat)

    val_sec = data_dict['data_loader']['data']['val']
    add_to_dict_if_exists(val_sec, 'num_bags',
                          config_parser, 'DATA_LOADER', 'VAL_NUM_BAGS', config_parser.getint)
    add_to_dict_if_exists(val_sec, 'mean_instances_per_bag',
                          config_parser, 'DATA_LOADER', 'VAL_MEAN_INSTANCES_PER_PAG', config_parser.getint)
    add_to_dict_if_exists(val_sec, 'std_instances_per_bag',
                          config_parser, 'DATA_LOADER', 'VAL_STD_INSTANCES_PER_BAG', config_parser.getfloat)

    if len(train_sec) == 0:
        del data_dict['data_loader']['data']['train']

    if len(val_sec) == 0:
        del data_dict['data_loader']['data']['val']

    if len(data_dict['data_loader']['data']) == 0:
        del data_dict['data_loader']['data']

    parsed_dict['data_settings'] = data_dict

    return parsed_dict


def parse_arguments(config_file_path: str = None) -> Dict:
    parser = argparse.ArgumentParser(description='Dynamic model argument parser.')
    parser.add_argument('-c', '--config_file', type=str,
                        help='path/to/configuration/file.ini')

    args, rest = parser.parse_known_args()

    config_file_path = args.config_file if args.config_file else config_file_path

    parsed_dict = create_config_dict(config_file_path=config_file_path)

    return parsed_dict


def parse_optuna_parameters(config_file_path: str = None) -> Dict:
    parser = argparse.ArgumentParser(description='Optuna argument parser.')
    parser.add_argument('-c', '--config_file', type=str,
                        help='path/to/configuration/file.ini')

    args, _ = parser.parse_known_args()

    # resolve all arguments and save it into the dict
    config_file_path = args.config_file if args.config_file else config_file_path

    parsed_dict = create_config_dict(config_file_path=config_file_path)

    return parsed_dict


def map_data_function(type_name: str) -> torch.nn.Module or None:
    """
    This function is used to map the config data types to a loader function. E. g. pass 'image' to get the function
    load_img_by_path as a return value. All return functions need a 'path' parameter, which is the path of a file from
    the corresponding annotation json.
    Extend the function according to your data loading needs.
    :param type_name: Name defined in your config file, e. g. image.
    :return: Tensor
    """
    if type_name == 'mnist_image':
        return ProcessMnistImage()
    elif type_name == 'svhn_image':
        return ProcessSVHNImage()
    elif type_name == 'mnist_audio':
        return ProcessAudioMnist()
    else:
        return lambda path: None


class ProcessMnistImage(torch.nn.Module):
    def __call__(self, path: str) -> torch.Tensor:
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))]
        )(Image.open(fp=path, mode='r'))


class ProcessSVHNImage(torch.nn.Module):
    def __call__(self, path: str) -> torch.Tensor:
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )(Image.open(fp=path, mode='r'))


class ProcessAudioMnist(torch.nn.Module):
    def __call__(self, path: str) -> torch.Tensor:
        data, _ = torchaudio.load(path)
        # make sure that data is normalized between -1.0 and 1.0!
        # This is sometimes dependent on your operation system!
        # data, _, _ = normalize(data, minimum=-1.0, maximum=1.0)
        return Sequential(
            # d = dimension (1) (s n) = sequence (e. g. 8000 (sample rate))
            Rearrange('d (n s) -> (d s) n', s=64)
        )(data)
