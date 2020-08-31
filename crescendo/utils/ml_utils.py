#!/usr/bin/env python3

import numpy as np
import random
import torch

from crescendo.utils.logger import logger_default as dlog

CUDA_AVAIL = torch.cuda.is_available()


def init_cuda(arg, cuda_avail=CUDA_AVAIL):
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
    average spread over all the features."""

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
    be cached as the model trains."""

    def __init__(self, root):
        self.root = root
        self.fname = f"{self.root}/info.txt"
        self.train_loss_list = []
        self.valid_loss_list = []
        self.learning_rates = []

    def step(self, epoch, t_loss, v_loss, lr, elapsed):
        """Writes the losses to disk and logs the information."""

        with open(self.fname, "w") as f:
            f.write(
                f"{epoch}\t{t_loss:.02e}\t{v_loss:.02e}\t{lr:.02e}\t"
                f"{elapsed:.02f}\n"
            )

        dlog.info(f"Epoch {epoch:04} [{elapsed:3.02f}s]")
        dlog.info(f'\tTrain Loss: {t_loss:.02e}')
        dlog.info(f"Eval complete: loss {v_loss:.02e}")

        self.train_loss_list.append(t_loss)
        self.valid_loss_list.append(v_loss)
        self.learning_rates.append(lr)
