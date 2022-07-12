#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Organization  : Technische Hochschule Ingolstadt
# Institution   : Computer Vision for Intelligent Mobility Systems (CVIMS)
# Created By    : Dominik Rößle
# =============================================================================
"""
This file downloads the house numbers dataset and provides a json description file.
"""
# =============================================================================
# Imports
# =============================================================================
import logging

import wget
import os
from PIL import Image
from tqdm import tqdm
from scipy.io import loadmat
from utils import create_directory, check_file_exists, create_json_file

current_file_directory = os.path.dirname(os.path.abspath(__file__))
default_dataset_path = os.path.abspath(os.path.join(current_file_directory, '..', 'datasets', 'svhn'))

svhn_url = r'http://ufldl.stanford.edu/housenumbers'
train_dataset_file_name = 'train_32x32.mat'
test_dataset_file_name = 'test_32x32.mat'


def download_svhn(save_path: str = default_dataset_path):
    """
    Uses the lecun urls to download the ubyte mnist files
    :param save_path: Directory to store ubyte files.
    :return:
    """
    dir_available = create_directory(save_path)
    if not dir_available:
        logging.error('SVHN download skipped due to save directory errors.')
        return

    logging.info('Starting download for street view house number data!')

    if not os.path.isfile(os.path.join(save_path, train_dataset_file_name)):
        wget.download('/'.join([svhn_url, train_dataset_file_name]), save_path)
    if not os.path.isfile(os.path.join(save_path, test_dataset_file_name)):
        wget.download('/'.join([svhn_url, test_dataset_file_name]), save_path)

    logging.info('Finished downloading street view house number data!')


def make_data_consistent(save_path: str = default_dataset_path):
    """
    Unpacks mat files and creates street view house number images
    :param save_path:
    :return: True or False for success or failure, respectively.
    """
    # check if every data file exists
    train_images_exist = check_file_exists(directory=save_path, full_file_name=train_dataset_file_name)
    test_images_exist = check_file_exists(directory=save_path, full_file_name=test_dataset_file_name)

    if not train_images_exist or not test_images_exist:
        download_svhn()

    def load_data(mat_file_path: str):
        data = loadmat(mat_file_path)
        # convert from HWCN into NHWC
        # H = height, W = width, C = channels, N = Number of images
        all_images = data['X'].transpose([3, 0, 1, 2])
        labels = data['y'].squeeze()
        # convert label 10 to 0 (to preserve the overall schema)
        labels[labels == 10] = 0

        assert(len(all_images) == len(labels))

        return all_images, labels

    for types in [('train', train_dataset_file_name),
                  ('test', test_dataset_file_name)]:
        dataset_type, data_file_name = types[0], types[1]
        counter = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        path = os.path.join(save_path, dataset_type)
        for key in counter:
            create_directory(os.path.join(path, str(key)), exists_ok=True)

        images, annotations = load_data(mat_file_path=os.path.join(save_path, data_file_name))

        for img, label in tqdm(zip(images, annotations), total=images.shape[0]):
            image_id = counter[label]
            full_img_path = os.path.join(path, str(label), str(image_id) + '.jpg')
            if os.path.isfile(full_img_path):
                # skip
                continue

            im = Image.fromarray(img)
            im.save(full_img_path)
            counter[label] += 1


def clean_intermediate_files(directory: str = default_dataset_path):
    # search for mat files
    logging.info('Cleaning the mat svhn files.')

    train_img_path = os.path.join(directory, train_dataset_file_name)
    if os.path.isfile(train_img_path):
        os.remove(train_img_path)
    train_label_path = os.path.join(directory, test_dataset_file_name)
    if os.path.isfile(train_label_path):
        os.remove(train_label_path)

    logging.info('Cleaning svhn mat files finished!')


def prepare_svhn_data(save_path: str = default_dataset_path, json_file_name='annotations'):
    make_data_consistent(save_path=save_path)
    create_json_file(data_name="svhn", save_path=save_path, file_name=json_file_name)
    # clean_intermediate_files(directory=save_path)


if __name__ == '__main__':
    prepare_svhn_data()
