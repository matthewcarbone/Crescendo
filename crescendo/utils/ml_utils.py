#!/usr/bin/env python3

import numpy as np
from itertools import product
import os
import pickle
import random
import subprocess
import torch
import yaml

from crescendo.utils.logger import logger_default as dlog
from crescendo.defaults import P_PROTOCOL


def read_config(path):
    """Reads the yaml ML config file.

    Parameters
    ----------
    path : str

    Returns
    -------
    dict
    """

    return yaml.safe_load(open(path))


def save_caches(protocol, mlds, data_loaders):
    """Pickles the cache results from every split to disk."""

    root = protocol.root
    epoch = protocol.epoch

    train_cache = protocol.eval(
        target_metadata=mlds.target_metadata,
        loader_override=data_loaders['train']
    )
    d = f"{root}/train"
    f = f"{d}/train_{epoch:04}.pkl"
    os.makedirs(d, exist_ok=True)
    pickle.dump(train_cache, open(f, 'wb'), protocol=P_PROTOCOL)

    valid_cache = protocol.eval(
        target_metadata=mlds.target_metadata,
        loader_override=data_loaders['valid']
    )
    d = f"{root}/valid"
    f = f"{d}/valid_{epoch:04}.pkl"
    os.makedirs(d, exist_ok=True)
    pickle.dump(valid_cache, open(f, 'wb'), protocol=P_PROTOCOL)

    test_cache = protocol.eval(
        target_metadata=mlds.target_metadata,
        loader_override=data_loaders['test']
    )
    d = f"{root}/test"
    f = f"{d}/====test_{epoch:04}====.pkl"
    os.makedirs(d, exist_ok=True)
    pickle.dump(test_cache, open(f, 'wb'), protocol=P_PROTOCOL)


def execution_parameters_permutations(dictionary):
    """Inputs a dictionary of a format such as

    eg = {
        hp1: [1, 2]
        hp2: [3, 4]
    }

    and returns a list of all permutations:

    eg1 = {
        hp1: 1
        hp2: 3
    }

    eg2 = {
        hp1: 1
        hp2: 4
    }

    eg3 = {
        hp1: 2
        hp2: 3
    }

    eg4 = {
        hp1: 2
        hp2: 4
    }
    """

    return [
        dict(zip(dictionary, prod)) for prod in product(
            *(dictionary[ii] for ii in dictionary)
        )
    ]


def _call_subprocess(script):
    process = subprocess.Popen(
        script, shell=True, stdout=subprocess.PIPE, universal_newlines=True
    )
    process.wait()


def init_cuda(arg, cuda_avail=torch.cuda.is_available()):
    """This is a critical helper function that allows seamless swapping
    between CPUs and GPUs. If CUDA is available, this function will apply the
    cuda() method to the object, essentially sending it to the GPU. If CUDA
    is not available, it simply does nothing."""

    if arg is None:
        return None
    if cuda_avail:
        return arg.cuda()
    return arg


def seed_all(seed):
    """Helper function that seeds the random, numpy and torch modules all
    at once."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def epoch_time(start_time, end_time):
    """Measures the elapsed minutes and seconds."""

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60.0)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60.0))
    return elapsed_mins, elapsed_secs


def mean_and_std(dat):
    """Takes a list of N data points, each with M features, and computes the
    mean and spread over the data points. Also logs the average mean and
    average spread over all the features.

    Parameters
    ----------
    dat : list
        The list of data to be turned into a numpy array. Contains training
        examples at the first level of depth, and features at the second level
        of depth.

    Returns
    -------
    np.ndarray, np.ndarray
        The mean and standard deviations of the features, averaged over all
        examples in dat.
    """

    dat = np.array(dat)
    mean = dat.mean(axis=0)
    sd = dat.std(axis=0)
    dlog.info(
        "Mean/sd of target data is "
        f"{mean.mean():.02e} +/- {sd.mean():.02e}"
    )
    return mean, sd


class Meter:
    """Used for keeping track of losses and other information that needs to
    be cached as the model trains.

    Attributes
    ----------
    root : str
        The location to which to save the results as produced by Meter. For
        example, training information will be saved to root/info.txt.
    fname : str
        The file name for saving information, e.g. root/info.txt.
    train_loss_list, valid_loss_list, learning_rates : list
        Lists containing various training information.
    """

    def __init__(self, root):
        self.root = root
        self.fname = f"{self.root}/info.txt"
        self.train_loss_list = []
        self.valid_loss_list = []
        self.learning_rates = []

    def step(self, epoch, t_loss, v_loss, lr, elapsed):
        """Writes the losses to disk and logs the information - i.e. steps the
        Meter.

        Parameters
        ----------
        epoch : int
            The current epoch.
        t_loss, v_loss, lr, elapsed : float
            Properties of training to be saved to disk: the training loss,
            validation loss, current learning rate, and time elapsed for
            training on that epoch.
        """

        with open(self.fname, "a") as f:
            f.write(
                f"{epoch:05}\t{t_loss:.04e}\t{v_loss:.04e}\t{lr:.04e}\t"
                f"{elapsed:.02f}\n"
            )

        dlog.info(f"Epoch {epoch:05} [{elapsed:3.02f}s]")
        dlog.info(f'\tTrain Loss: {t_loss:.03e}')
        dlog.info(f"\tValid loss {v_loss:.03e}")

        self.train_loss_list.append(t_loss)
        self.valid_loss_list.append(v_loss)
        self.learning_rates.append(lr)
