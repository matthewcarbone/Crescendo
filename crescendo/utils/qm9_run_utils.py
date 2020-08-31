#!/usr/bin/env python3

from itertools import product
import os
import pickle
import random
import uuid
import yaml

from crescendo.protocols.graph_protocols import GraphToVectorProtocol
from crescendo.datasets.qm9 import QM9GraphDataset
from crescendo.utils.py_utils import check_for_environment_variable
from crescendo.defaults import P_PROTOCOL, QM9_DS_ENV_VAR
from crescendo.utils.logger import logger_default as dlog


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
        meta=mlds.target_metadata,
        loader_override=data_loaders['train']
    )
    d = f"{root}/train"
    f = f"{d}/train_{epoch:04}.pkl"
    os.makedirs(d, exist_ok=True)
    pickle.dump(train_cache, open(f, 'wb'), protocol=P_PROTOCOL)

    valid_cache = protocol.eval(
        meta=mlds.target_metadata,
        loader_override=data_loaders['valid']
    )
    d = f"{root}/valid"
    f = f"{d}/valid_{epoch:04}.pkl"
    os.makedirs(d, exist_ok=True)
    pickle.dump(valid_cache, open(f, 'wb'), protocol=P_PROTOCOL)

    test_cache = protocol.eval(
        meta=mlds.target_metadata,
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


class Manager:
    """Helps keep track of the different trials being performed, and writes
    important information to disk.

    Parameters
    ----------
    dsname : str
        The name of the dataset which must match that of the previously loaded
        and configured datasets.
    directory : str
        The location of the cache directory
    """

    def __init__(
        self, dsname, directory=check_for_environment_variable(QM9_DS_ENV_VAR)
    ):
        # Location of the directory containing the datasets
        self.root_above = f"{directory}/{dsname}"

    def prime(self, config_path='config.yaml', max_hp=24):
        """Primes the computation by creating the necessary trial directories
        and parsing the configurations into the right places. This is
        essentially the setup to hyperparameter tuning, if desired, or running
        many different combinations of hyperparameters.

        Parameters
        ----------
        config_path : str
            The absolute path to the configuration file used to setup the
            trials.
        max_hp : int
            The maximum number of trials to generate at once. If the number of
            total hyperparameter combinations is more than max_hp, they will
            be selected randomly from the permutations.
        """

        dlog.info("Priming machine learning combinations")
        config = read_config(config_path)
        combinations = execution_parameters_permutations(config)
        dlog.info(f"Total of {len(combinations)} created")
        if len(combinations) > max_hp:
            combinations = random.sample(combinations, max_hp)
            dlog.warning(
                f"Length of all combinations exceeds max_hp of {max_hp} - "
                f"new length is {len(combinations)}"
            )

        cc = 0
        for combo in combinations:
            d = f"{self.root_above}/{cc:03}"
            os.makedirs(d)
            path = f"{d}/config.yaml"
            with open(path, 'w') as f:
                yaml.dump(combo, f, default_flow_style=False)
            cc += 1


def run_single_protocol(args, config, trial=str(uuid.uuid4())):
    """Initializes a machine learning protocol from a dictionary of
    parameters.

    Parameters
    ----------
    config : dict
        Must have a 1-to-1 correspondence between keys and ML parameters.
    args
        An argparse-parsed arguments object.
    trial : str
        Defaults to a random hash if unspecified.
    """

    mlds = QM9GraphDataset(args.train)
    data_loaders = mlds.get_loaders()

    protocol = GraphToVectorProtocol(
        args.train, trial,
        trainLoader=data_loaders['train'],
        validLoader=data_loaders['valid']
    )

    protocol.initialize_model(
        n_node_features=mlds.node_edge_features[0],
        n_edge_features=mlds.node_edge_features[1],
        output_size=mlds.n_targets,
        hidden_node_size=config['hidden_node_size'],
        hidden_edge_size=config['hidden_edge_size']
    )

    protocol.initialize_support(
        optimizer=(
            config['optimizer'], {
                'lr': config['lr']
            }
        ),
        scheduler=(
            'rlrp', {
                'patience': config['patience'],
                'factor': config['factor'],
                'min_lr': config['min_lr']
            }
        )
    )

    protocol.train(config['epochs'], clip=config['clip'])
    save_caches(protocol, mlds, data_loaders)
