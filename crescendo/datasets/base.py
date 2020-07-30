#!/usr/bin/env python3

import torch

import torch.utils.data


class _BaseCore(torch.utils.data.Dataset):

    def __init__(self):
        self.raw = None
        self.ml_data = None

    def __getitem__(self, index):
        return self.ml_data[index]

    def __len__(self):
        return len(self.ml_data)

    def load(self):
        raise NotImplementedError

    def init_ml(self):
        raise NotImplementedError
