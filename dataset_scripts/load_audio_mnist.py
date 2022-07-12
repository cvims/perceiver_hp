#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Organization  : Technische Hochschule Ingolstadt
# Institution   : Computer Vision for Intelligent Mobility Systems (CVIMS)
# Created By    : Dominik Rößle
# =============================================================================
"""
This file uses the data downloaded from https://github.com/soerenab/AudioMNIST/blob/master/preprocess_data.py
preprocesses them and saves them into a new format (hdf5).
"""
# =============================================================================
# Imports
# =============================================================================
import logging
import os
from tqdm import tqdm
from utils import create_directory, create_json_file
import scipy.io.wavfile as wav
import librosa
import numpy as np
np.random.seed(42)
import random
random.seed(42)
import shutil
import svn.remote

current_file_directory = os.path.dirname(os.path.abspath(__file__))
default_dataset_path = os.path.abspath(os.path.join(current_file_directory, '..', 'datasets', 'audio_mnist'))

audio_mnist_url = r'https://github.com/soerenab/AudioMNIST.git/trunk/data'

dataset_folder_name = 'dataset'
file_format = 'wav'


def download_audio_mnist(save_path: str = default_dataset_path):
    r = svn.remote.RemoteClient(audio_mnist_url)
    print('Downloading audio mnist data.')
    export_path = os.path.join(save_path, dataset_folder_name)
    r.export(export_path)
    print('Finished downloading audio mnist data.')


def preprocess_data(data_info):
    data_embed = {k: [] for k in data_info}
    for digit in data_info:
        for file_path in tqdm(data_info[digit], desc=f'Preprocess audio data for digit {digit}'):
            fs, data = wav.read(file_path)
            data = librosa.core.resample(y=data.astype(np.float32), orig_sr=fs, target_sr=8000, res_type="scipy")
            # zero padding
            if len(data) > 8000:
                raise ValueError("data length cannot exceed padding length.")
            if len(data) < 8000:
                embedded_data = np.zeros(8000)
                offset = np.random.randint(low=0, high=8000 - len(data))
                embedded_data[offset:offset + len(data)] = data
            elif len(data) == 8000:
                # nothing to do here
                embedded_data = data
                pass

            data_embed[digit].append(embedded_data)
    return data_embed


def make_data_consistent(save_path: str = default_dataset_path):
    """

    :param save_path:
    :return:
    """
    def load_data_information(file_path: str):
        # collect all files

        def find_files_recursive(start_path):
            found_files = []

            for entry in os.listdir(start_path):
                current_path = os.path.join(start_path, entry)
                if os.path.isdir(current_path):
                    new_files = find_files_recursive(start_path=current_path)
                    found_files.extend(new_files)

                if os.path.isfile(current_path):
                    if file_format:
                        if current_path.split('.')[-1] != file_format:
                            continue

                    found_files.append(current_path)

            return found_files

        # apply to the correct numbers
        information = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

        file_paths = find_files_recursive(file_path)

        # sort the file paths
        for f_path in tqdm(file_paths):
            # the audio file always starts with the corresponding spoken digit
            file_name = os.path.basename(f_path)
            digit = file_name.split('_')[0]
            information[int(digit)].append(f_path)

        return information

    def create_train_test_split(data_info, ratio=0.8):
        train_info = {}
        test_info = {}
        for info in data_info:
            paths = data_info[info]
            path_len = len(paths)
            random.shuffle(paths)
            train_paths_len = round(path_len * ratio)
            train_info[info] = paths[:train_paths_len]
            test_info[info] = paths[train_paths_len:]

        return train_info, test_info

    def write_to(to_path: str, data_info):
        counter = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        for key in counter:
            create_directory(os.path.join(to_path, str(key)), exists_ok=True)

        for digit in data_info:
            identifier = 0
            for data in data_info[digit]:
                new_file_path = os.path.join(to_path, str(digit), str('.'.join((str(identifier), file_format))))
                wav.write(filename=new_file_path, rate=48000, data=data)
                identifier += 1

    def copy_to(to_path: str, data_info):
        counter = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        for key in counter:
            create_directory(os.path.join(to_path, str(key)), exists_ok=True)

        logging.info(f'Copying data to {to_path}')
        for info in data_info:
            identifier = 0
            for path in tqdm(data_info[info], desc=f'Copying data for digit: {info}'):
                extension = os.path.splitext(path)[1]
                shutil.copy(path, os.path.join(to_path, str(info), str(identifier) + extension))
                identifier += 1

    if not os.path.isdir(os.path.join(save_path, dataset_folder_name)):
        download_audio_mnist(save_path=save_path)

    data_information = load_data_information(file_path=os.path.join(default_dataset_path, dataset_folder_name))
    preprocessed_data = preprocess_data(data_info=data_information)
    train_information, test_information = create_train_test_split(data_info=preprocessed_data)
    write_to(to_path=os.path.join(save_path, 'train'), data_info=train_information)
    write_to(to_path=os.path.join(save_path, 'test'), data_info=test_information)


def prepare_audio_mnist_data(save_path: str = default_dataset_path, json_file_name='annotations'):
    make_data_consistent(save_path=save_path)
    create_json_file(data_name="audio_mnist", save_path=save_path, file_name=json_file_name)


if __name__ == '__main__':
    prepare_audio_mnist_data()
