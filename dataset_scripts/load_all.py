#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Organization  : Technische Hochschule Ingolstadt
# Institution   : Computer Vision for Intelligent Mobility Systems (CVIMS)
# Created By    : Dominik Rößle
# =============================================================================
"""
This file uses all load dataset scripts and executes them sequentially.
"""
# =============================================================================
# Imports
# =============================================================================
import os
import argparse
from load_mnist import prepare_mnist_data
from load_housenumbers import prepare_svhn_data
from load_audio_mnist import prepare_audio_mnist_data
from utils import create_train_val_split

current_file_directory = os.path.dirname(os.path.abspath(__file__))
default_dataset_path = os.path.abspath(os.path.join(current_file_directory, '..', 'datasets'))


def load_and_prepare_datasets(save_dir: str = default_dataset_path, val_percentage=20):
    # creates training and test splits already
    prepare_audio_mnist_data(save_path=os.path.join(save_dir, 'audio_mnist'))
    # create train val split audio mnist from the created train split of the above step
    create_train_val_split(
        annotation_file_path=os.path.join(save_dir, 'audio_mnist', 'train', 'annotations.json'),
        output_dir=os.path.join(save_dir, 'audio_mnist'),
        val_percentage=val_percentage)

    # creates training and test splits already
    prepare_mnist_data(save_path=os.path.join(save_dir, 'mnist'))
    # create train val split mnist from the created train split of the above step
    create_train_val_split(
        annotation_file_path=os.path.join(save_dir, 'mnist', 'train', 'annotations.json'),
        output_dir=os.path.join(save_dir, 'mnist'),
        val_percentage=val_percentage)

    # creates training and test splits already
    prepare_svhn_data(save_path=os.path.join(save_dir, 'svhn'))
    # create train val split svhn from the created train split of the above step
    create_train_val_split(
        annotation_file_path=os.path.join(save_dir, 'svhn', 'train', 'annotations.json'),
        output_dir=os.path.join(save_dir, 'svhn'),
        val_percentage=val_percentage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load and prepare the datasets.')
    parser.add_argument('-s', '--save-dir', type=str, default=default_dataset_path,
                        help='Path/to/the/save/directory')

    args = parser.parse_args()

    load_and_prepare_datasets(save_dir=args.save_dir, val_percentage=20)
