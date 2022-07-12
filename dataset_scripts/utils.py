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
import copy
import os
import logging
import json
import random


def create_directory(directory: str, exists_ok=True):
    """
    Checks the directory and creates it if it does not exist.
    :param directory: lookup directory
    :param exists_ok: Only used for error message and return value
    if the directory exists and is not empty
    :return: True or False
    """
    if os.path.isdir(directory):
        # check if it is empty
        if os.listdir(directory):
            if not exists_ok:
                logging.error(f'The directory is not empty: {directory}.')
                return False
            else:
                logging.info(f'The directory is not empty: {directory}.')
                return True
        else:
            logging.info(f'Directory already exists: {directory}')
            return True
    else:
        os.makedirs(directory)
        logging.info(f'Directory created: {directory}')
        return True


def check_file_exists(directory: str, full_file_name: str):
    """
    Checks if the file is is existent.
    :param directory: Directory where the file is located.
    :param full_file_name: Full file name with its extension.
    :return: True or False
    """
    if os.path.isdir(directory):
        if os.path.isfile(os.path.join(directory, full_file_name)):
            return True
        else:
            return False
    else:
        return False


def create_json_file(data_name: str, save_path: str, file_name: str = 'annotations', recreate: bool = False):
    """
        Iterates the save_path and creates a simple json annotation file.
        :param data_name: Custom name of the dataset.
        :param save_path: Lookup path for mnist data
        :param file_name: Json file name
        :param recreate: If the annotation file already exists. True deletes the current one and
        creates a new one and False is skipping the annotation creation for existing files.
        :return:
        """
    logging.info('Creating annotation files!')
    # save train json file
    if '.json' not in file_name:
        file_name = file_name + '.json'

    # create json for train and test data
    for dataset_type in ['train', 'test']:
        path = os.path.join(save_path, dataset_type)
        ann_path = os.path.join(path, file_name)
        if os.path.isfile(ann_path):
            if recreate:
                logging.warning(f'Deleting annotation file {ann_path}. ')
                os.remove(ann_path)
            else:
                logging.warning(f'Skipping to recreate annotation file {ann_path}. I found an already existing '
                                f'annotation file.')
                continue

        base_json = get_json_base_file_format()

        for dir_name in os.listdir(path):
            full_dir_path = os.path.join(path, dir_name)
            if os.path.isdir(full_dir_path):
                for img_file in os.listdir(full_dir_path):
                    full_img_path = os.path.join(full_dir_path, img_file)
                    if not os.path.isdir(full_img_path):
                        img_name, img_extension = os.path.splitext(img_file)
                        # save the annotation
                        annotation = get_json_descriptive_format(_id=int(img_name),
                                                                 file_name=img_file,
                                                                 full_file_path=full_img_path)
                        base_json[int(dir_name)].append(annotation)

        with open(os.path.join(path, file_name), 'w') as f:
            logging.info(f'Saving json file {os.path.join(path, file_name)}')
            base_json['name'] = data_name
            json.dump(base_json, f)

    logging.info('Finished creating annotation files!')


def get_json_base_file_format():
    json_format = {
        "name": "",
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [],
        9: []
    }

    return json_format


def get_json_descriptive_format(_id: int, file_name: str, full_file_path: str):
    """
    Inner part of base json file format, e. g. inside the array of 0: []
    :param _id: Unique identifier
    :param file_name: File name with extension
    :param full_file_path: Full absolute file path
    :return:
    """
    json_format = {
        "id": _id,
        "file_name": file_name,
        "full_file_path": full_file_path
    }

    return json_format


def create_train_val_split(annotation_file_path, output_dir, val_percentage=20, seed=999):
    """
    This method creates new annotation file splits. So dont delete the main files because the new annotation files
    point to the path references. The splits are always stratified.
    :param annotation_file_path: Annotation file path to split from including the file_name and extension.
    :param output_dir: Directory to save new splits
    :param val_percentage: percentage of validation split [1, 99[
    :param seed: Random seed
    """

    assert os.path.isfile(annotation_file_path)
    assert 0 < val_percentage < 100

    random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, 'train_annotations.json')
    val_path = os.path.join(output_dir, 'val_annotations.json')

    with open(os.path.join(annotation_file_path), 'r') as f:
        logging.info(f'Reading from json file {annotation_file_path}')
        json_file = json.load(f)
        name = json_file['name']

        compl_len = 0
        compl_train_len = 0

        train_json_file = {
            "name": name
        }

        val_json_file = {
            "name": name
        }

        for num in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            num_section = copy.deepcopy(json_file[num])
            random.shuffle(num_section)
            all_len = len(num_section)
            compl_len += all_len
            train_len = int(all_len * ((100 - val_percentage) / 100))
            compl_train_len += train_len

            # random from every single list
            train_split = num_section[:train_len]
            val_split = num_section[train_len:]

            train_json_file[str(num)] = train_split
            val_json_file[str(num)] = val_split

    print(f'Data split completed for directory {output_dir}')
    print(f'Splitted training examples ({compl_train_len}/{compl_len})')
    print(f'Splitted validation examples ({compl_len-compl_train_len}/{compl_len})')

    for split, json_file in [(train_path, train_json_file), (val_path, val_json_file)]:
        with open(split, 'w') as f:
            logging.info(f'Saving json file {split}')
            json.dump(json_file, f)
