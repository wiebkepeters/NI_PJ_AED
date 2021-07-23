# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from typing import Dict
# from torch._C import dtype, float32, int32

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from ._tut_sed_synthetic_2016 import TUTSEDSynthetic2016

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['get_tut_sed_data_loader']


def custom_collate_fn(batch):

    seq_length = 256
    data = []
    labels = []
    lengths = []

    shortes_length = float('inf')

    for item in batch:
        d, l = preprocess(item[0], item[1], seq_length)
        data.append(d)
        labels.append(l)
        s = d.shape[0]
        lengths.append(s)

        if s < shortes_length:
            shortes_length = s

    # data = sorted(data, key=lambda d:d.shape[0], reverse=True)
    # labels = sorted(labels, key=lambda l:l.shape[0], reverse=True)
    # lengths = sorted(lengths, reverse=True)
    # # lengths = [l - shortes_length for l in lengths]

    data = pad_sequence(data, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)

    return [data, labels, lengths, seq_length, shortes_length]


def preprocess(data, labels, seq_length):
    return data[:-(len(data) % seq_length), :], labels[:-(len(labels) % seq_length)]



def get_tut_sed_data_loader(root_dir: str,
                            split: str,
                            batch_size: int,
                            shuffle: bool,
                            drop_last: bool,
                            input_features_file_name: str,
                            target_values_input_name: str) \
        -> DataLoader:
    """Creates and returns the data loader.

    :param root_dir: The root dir for the dataset.
    :type root_dir: str
    :param split: The split of the data (training, \
                          validation, or testing).
    :type split: str
    :param batch_size: The batch size.
    :type batch_size: int
    :param shuffle: Shuffle the data?
    :type shuffle: bool
    :param drop_last: Drop last examples?
    :type drop_last: bool
    :param input_features_file_name: Input features file nsame.
    :type input_features_file_name: str
    :param target_values_input_name: Target values file name.
    :type target_values_input_name: str
    :return: Data loader.
    :rtype: torch.utils.data.DataLoader
    """
    data_loader_kwargs: Dict = {
        'root_dir': root_dir,
        'split': split,
        'input_features_file_name': input_features_file_name,
        'target_values_input_name': target_values_input_name}

    dataset = TUTSEDSynthetic2016(**data_loader_kwargs)

    return DataLoader(
        dataset=dataset, batch_size=batch_size,
        shuffle=shuffle if split == 'training' else False,
        drop_last=drop_last,
        collate_fn=custom_collate_fn)

# EOF


