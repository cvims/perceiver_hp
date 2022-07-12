#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Organization  : Technische Hochschule Ingolstadt
# Institution   : Computer Vision for Intelligent Mobility Systems (CVIMS)
# Created By    : Dominik Rößle
# =============================================================================
"""
This file downloads the mnist dataset and provides a json description file.
"""
# =============================================================================
# Imports
# =============================================================================
import logging

import wget
import os
import gzip
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import create_directory, check_file_exists, create_json_file

current_file_directory = os.path.dirname(os.path.abspath(__file__))
default_dataset_path = os.path.abspath(os.path.join(current_file_directory, '..', 'datasets', 'mnist'))

mnist_url = r'http://yann.lecun.com/exdb/mnist'
train_images_url_file_name = 'train-images-idx3-ubyte.gz'
train_labels_url_file_name = 'train-labels-idx1-ubyte.gz'
test_images_url_file_name = 't10k-images-idx3-ubyte.gz'
test_labels_url_file_name = 't10k-labels-idx1-ubyte.gz'


def download_mnist(save_path: str = default_dataset_path):
    """
    Uses the lecun urls to download the ubyte mnist files
    :param save_path: Directory to store ubyte files.
    :return:
    """
    dir_available = create_directory(save_path)
    if not dir_available:
        logging.error('Mnist download skipped due to save directory errors.')
        return

    logging.info('Starting download for mnist data!')

    if not os.path.isfile(os.path.join(save_path, train_images_url_file_name)):
        wget.download('/'.join([mnist_url, train_images_url_file_name]), save_path)
    if not os.path.isfile(os.path.join(save_path, train_labels_url_file_name)):
        wget.download('/'.join([mnist_url, train_labels_url_file_name]), save_path)
    if not os.path.isfile(os.path.join(save_path, test_images_url_file_name)):
        wget.download('/'.join([mnist_url, test_images_url_file_name]), save_path)
    if not os.path.isfile(os.path.join(save_path, test_labels_url_file_name)):
        wget.download('/'.join([mnist_url, test_labels_url_file_name]), save_path)

    logging.info('Finished downloading mnist data!')


def make_data_consistent(save_path: str = default_dataset_path):
    """
    Unpacks ubyte files and creates mnist images
    :param save_path:
    :return:
    """
    # check if every data file exists
    train_images_exist = check_file_exists(directory=save_path, full_file_name=train_images_url_file_name)
    train_labels_exist = check_file_exists(directory=save_path, full_file_name=train_labels_url_file_name)
    test_images_exist = check_file_exists(directory=save_path, full_file_name=test_images_url_file_name)
    test_labels_exist = check_file_exists(directory=save_path, full_file_name=test_labels_url_file_name)

    if not train_images_exist or not train_labels_exist or not test_images_exist or not test_labels_exist:
        download_mnist(save_path=save_path)

    def load_images(gz_file_path: str):
        with gzip.open(filename=gz_file_path, mode='r') as b:
            # The byte string is provided as follows:
            # First four bytes is a magic number
            # Second four bytes is the image amount
            # Third four bytes is the row count
            # Fourth four bytes is the column count
            _ = int.from_bytes(b.read(4), 'big')
            image_amount = int.from_bytes(b.read(4), 'big')
            row_count = int.from_bytes(b.read(4), 'big')
            column_count = int.from_bytes(b.read(4), 'big')
            # now read the image complete image data
            complete_image_data = b.read()
            # whe use numpy to read the buffer data
            # make sure you always use np.uint8 to preserve the pixel range 0-255
            all_images = np.frombuffer(complete_image_data, dtype=np.uint8)\
                .reshape((image_amount, row_count, column_count))
            return all_images

    def load_labels(gz_file_path: str):
        with gzip.open(filename=gz_file_path, mode='r') as b:
            # First four bytes is a magic number
            # Second four bytes is the label amount
            _ = int.from_bytes(b.read(4), 'big')
            label_amount = int.from_bytes(b.read(4), 'big')
            label_data = b.read()
            labels = np.frombuffer(label_data, dtype=np.uint8)

            return labels

    # make training and test data consistent
    for types in [('train', train_images_url_file_name, train_labels_url_file_name),
                  ('test', test_images_url_file_name, test_labels_url_file_name)]:
        dataset_type, image_file_name, label_file_name = types[0], types[1], types[2]
        counter = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        path = os.path.join(save_path, dataset_type)
        for key in counter:
            create_directory(os.path.join(path, str(key)), exists_ok=True)

        images = load_images(gz_file_path=os.path.join(save_path, image_file_name))
        annotations = load_labels(gz_file_path=os.path.join(save_path, label_file_name))

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
    # search for ubyte files
    logging.info('Cleaning the ubyte mnist files.')

    train_img_path = os.path.join(directory, train_images_url_file_name)
    if os.path.isfile(train_img_path):
        os.remove(train_img_path)
    train_label_path = os.path.join(directory, train_labels_url_file_name)
    if os.path.isfile(train_label_path):
        os.remove(train_label_path)
    test_img_path = os.path.join(directory, test_images_url_file_name)
    if os.path.isfile(test_img_path):
        os.remove(test_img_path)
    test_label_path = os.path.join(directory, test_labels_url_file_name)
    if os.path.isfile(test_label_path):
        os.remove(test_label_path)

    logging.info('Cleaning mnist ubyte files finished!')


def prepare_mnist_data(save_path: str = default_dataset_path, json_file_name='annotations'):
    make_data_consistent(save_path=save_path)
    create_json_file(data_name="mnist", save_path=save_path, file_name=json_file_name)
    # clean_intermediate_files(directory=save_path)


if __name__ == '__main__':
    prepare_mnist_data()
