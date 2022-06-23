# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: TXacs (txacs1993@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Parts of the code here are adapted from https://gist.github.com/branislav1991/4c143394bdad612883d148e0617bdccd

import torch
from torch.utils import data
import h5py
import numpy as np
import random
import os
from Merak import mpu
import json

def get_data_files(args, seed):
    files = [os.path.join(args.data_files, f) for f in os.listdir(args.data_files) if
            os.path.isfile(os.path.join(args.data_files, f)) and 'training' in f]
    files.sort()
    random.Random(seed).shuffle(files)

    start_idx = int(len(files) / mpu.get_data_parallel_world_size() * mpu.get_data_parallel_rank())
    end_idx = int(len(files) / mpu.get_data_parallel_world_size() * (mpu.get_data_parallel_rank() + 1))
    data_file = files[start_idx: end_idx]

    return data_file

class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed
    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)

class HDF5Dataset(data.Dataset):
    def __init__(self, args, seed, max_pred_length, json_file="./data_info.json", data_cache_size=1, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform
        self.max_pred_length = max_pred_length

        files = get_data_files(args, seed)
        self.num_files = len(files)
        self.file_idx = 0
        self.samples = 0
        self.total_samples = 0
        self.inputs = None
        self.iter_idx = None

        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')
        if not os.path.isfile(json_file) or os.path.getsize(json_file) < 10:
            for h5dataset_fp in files:
                # print(h5dataset_fp)
                # self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
                self._add_data_infos(h5dataset_fp)
            with open(json_file, 'w', encoding="utf-8") as f:
                json.dump(self.data_info, f)
        else:
            with open(json_file, 'r', encoding="utf-8") as f:
                self.data_info = json.loads(f.read())
            self.get_total_samples()



    def __getitem__(self, index):
        # get data
        self.samples = self.get_data_infos()[self.file_idx]['length']
        if self.inputs is None:
            self.inputs, self.iter_idx = self.get_data(self.file_idx)
        try:
            index = next(self.iter_idx)
        except StopIteration:
            self.file_idx += 1
            if self.file_idx >= self.num_files:
                self.file_idx = 0
            self.inputs, self.iter_idx = self.get_data(self.file_idx)

        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids,
                "masked_lm_labels": masked_lm_labels, "next_sentence_labels": next_sentence_labels}

    def __len__(self):
        return self.total_samples

    def _add_data_infos(self, file_path):
        with h5py.File(file_path) as h5_file:
            idx = -1
            self.data_info.append({'file_path': file_path, 'length': len(np.asarray(h5_file["input_ids"][:])), 'cache_idx': idx})
        self.get_total_samples()


    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
            inputs = [np.asarray(h5_file[key][:]) for key in keys]
            # add data to the data cache and retrieve
            # the cache index
            idx = self._add_to_cache(inputs, file_path)

            # find the beginning index of the hdf5 file we are looking for
            file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)

            # the data info should have the same index since we loaded it in the same way
            self.data_info[file_idx + idx]['cache_idx'] = idx


    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info]
        return data_info_type

    def get_data(self, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos()[i]['file_path']
        idx = list(range(self.get_data_infos()[i]['length']))
        random.shuffle(idx)

        if fp not in self.data_cache:
            self._load_data(fp)
        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos()[i]['cache_idx']
        return self.data_cache[fp][cache_idx], iter(idx)

    def get_total_samples(self):
        num_sample = []
        for data_info in self.data_info:
            num_sample.append(data_info['length'])
        self.total_samples = sum(num_sample)