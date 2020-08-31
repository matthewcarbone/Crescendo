#!/usr/bin/env python3

"""Module for loading in data from the QM9 database."""

import datetime
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import numpy as np
import os as os
import random

from dgllife.utils.analysis import summarize_a_mol
from dgllife.utils.mol_to_graph import mol_to_bigraph
import glob2
import pickle as pickle
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
import torch

from crescendo import defaults
from crescendo.samplers.base import Sampler as RandomSampler
from crescendo.utils.logger import logger_default as dlog
from crescendo.utils.py_utils import check_for_environment_variable, \
    intersection
from crescendo.readers.qm9_readers import parse_QM8_electronic_properties, \
    read_qm9_xyz
from crescendo.utils.ml_utils import mean_and_std
from crescendo.utils.mol_utils import all_analysis
from crescendo.utils.timing import time_func


class _SimpleLoadSaveOperations:

    def _save_state(
        self, class_type, directory=None, override=False
    ):
        """Dumps the class as a dictionary into a pickle file. It will save
        the object to the dsname directory as class_type.pkl. Specifically, it
        saves to directory/dsname/class_type.pkl. (class_type = machine
        learning dataset)."""

        if directory is None:
            directory = check_for_environment_variable(defaults.QM9_DS_ENV_VAR)

        full_dir = f"{directory}/{self.dsname}"
        full_path = f"{full_dir}/{class_type}.pkl"

        if os.path.exists(full_path) and not override:
            error = \
                f"Full path {full_path} exists and override is False - " \
                "exiting and not overwriting"
            dlog.error(error)
            return

        elif os.path.exists(full_path) and override:
            warning = \
                f"Full path {full_path} exists and override is True - " \
                "overwriting saved dataset"
            dlog.warning(warning)

        os.makedirs(full_dir, exist_ok=True)

        d = self.__dict__
        pickle.dump(d, open(full_path, 'wb'), protocol=defaults.P_PROTOCOL)

        dlog.info(f"Saved {full_path}")

    def _load_state(self, class_type, dsname, directory=None):
        """Reloads the dataset of the specified name and directory."""

        if directory is None:
            directory = check_for_environment_variable(defaults.QM9_DS_ENV_VAR)

        full_path = f"{directory}/{dsname}/{class_type}.pkl"
        d = pickle.load(open(full_path, 'rb'))

        for key, value in d.items():
            setattr(self, key, value)

        dlog.info(f"Loaded from {full_path}")

    def check_exists(self, class_type, directory=None):
        if directory is None:
            directory = check_for_environment_variable(defaults.QM9_DS_ENV_VAR)

        full_dir = f"{directory}/{self.dsname}"
        full_path = f"{full_dir}/{class_type}.pkl"
        if os.path.exists(full_path):
            return full_path
        return None


class QM9DataPoint:
    """Functions for determining structures from a smiles input using rdkit
    chem. Meant to be a standalone, python + numpy-only class with no other
    dependencies. Essentially a mutable named tuple.

    Attributes
    ----------
    qm9ID : int
        The QM9 identifier.
    smiles : tuple
        A 2-tuple in which smiles[0] is the standard SMILES string and
        smiles[1] is the canonicalized version.
    qm9properties : list, optional
        A list of the properties in the QM9 dataset, starting with the
        "A" rotational constant and ending with the "Cv" heat capacity.
    xyz : array_like, optional
        Usually a numpy array of atoms (each row is an atom) and geometries
        (each column, x, y, z).
    elements : list, optional
        A list of strings corresponding to the element of each atom in the xyz
        file. Note that this (and the xyz file) includes hydrogen).
    zwitter : bool, optional
        True if the element is a Zwitterion, and False otherwise.
    qm8properties : list, optional
        The electronic properties corresponding to the QM8 dataset.
    oxygenXANES, nitrogenXANES : list, optional
        The XANES spectrum for Oxygen and Nitrogen of the whole molecule. Note
        that the energy-axis information is the same for each (O or N) and is
        stored in the QM9Dataset class so as not to duplicate information
        needlessly.
    nheavy : int
        The number of heavy atoms in the molecule.
    """

    def __init__(
        self,
        qm9ID,
        smiles,
        qm9properties=None,
        xyz=None,
        elements=None,
        zwitter=None
    ):
        self.qm9ID = qm9ID
        self.smiles = smiles
        self.qm9properties = qm9properties
        self.xyz = xyz
        self.elements = elements
        self.zwitter = zwitter
        self.qm8properties = None
        self.oxygenXANES = None
        self.nitrogenXANES = None
        self.nheavy = sum([e != 'H' for e in self.elements])
        self.mol = None
        self.mw = None
        self.graph = None
        self.summary = None


class QM9Dataset(_SimpleLoadSaveOperations):
    """Container for the QM9 data. This is meant to be a standalone dataset
    that is only truly dependent on internal (crescendo) packages, pure python
    and numpy. The next iteration of this dataset is the QM9GraphDataset, which
    will depend on other packages including torch.

    Attributes
    ----------
    raw : dict
        A dictionary or raw data as constructed by the load method. In the case
        of QM9, this is simply the QM9SmilesDatum object keyed by the QM9 ID.
    qm8_electronic_properties : dict
        Default as None, this is initialized by the
        load_qm8_electronic_properties, which loads in the qm8 electronic
        properties from a specified path and stores them, again by QM9 ID.
    debug : int
        Default as -1, the debug flag simply indexes the max number of
        geometries to load. Will load all by default (indicated by -1).
    dt_created : datetime.datetime
        The time at which this dataset was initialized.

    Parameters
    ----------
    dsname : str, optional
        The user-defined name of the datset. If it is not specified, it
        will be set to "QM9_dataset_default"
    """

    def __init__(self, dsname=None, debug=-1):

        if dsname is None:
            dlog.warning(f"Dataset name initialized to default {dsname}")
            self.dsname = "QM9_dataset_default"
        else:
            self.dsname = dsname
        self.raw = dict()
        self.debug = debug
        self.dt_created = datetime.datetime.now()
        self.oxygenXANES_grid = None
        self.nitrogenXANES_grid = None

    def __getitem__(self, ii):
        try:
            res = self.raw[ii]
        except KeyError:
            critical = \
                f"QM9 ID {ii} does not exist in the database. " \
                "Note that the QM9Dataset accesses it's data via key/values " \
                "in a dictionary, and a correct QM9 ID must be specified."
            dlog.critical(critical)
            raise KeyError(critical)
        return res

    def __len__(self):
        return len(self.raw)

    def save_state(self, directory=None, override=False):
        """Dumps the class as a dictionary into a pickle file. It will save
        the object to the dsname directory as raw.pkl. Specifically, it saves
        to directory/dsname/raw.pkl."""

        self._save_state(
            class_type='raw', directory=directory, override=override
        )

    def load_state(self, dsname, directory=None):
        """Reloads the dataset of the specified name and directory."""

        self._load_state(class_type='raw', dsname=dsname, directory=directory)

    @time_func(dlog)
    def load(self, path=None, n_workers=cpu_count()):
        """Loads in **only** the QM9 raw data from .xyz files.

        Parameters
        ----------
        path : str, optional
            Path to the directory containing the qm9 .xyz files. For instance,
            if your xyz files are in directory /Users/me/data, then that should
            be the path. If path is None by default, it will check the
            os.environ dictionary for QM9_DATA_PATH, and if that does not
            exist, it will throw an error.
        n_workers : int
            The number of processes during loading. Defaults to the number of
            available CPU's on your machine.
        """

        if path is None:
            path = check_for_environment_variable(defaults.QM9_ENV_VAR)
        dlog.info(f"Loading QM9 from {path}")

        # Get a list of all of the paths of the xyz files
        all_xyz_paths = glob2.glob(path + "/*.xyz")
        total_xyz = len(all_xyz_paths)

        # Trim the total dataset if we're debugging and want to go fast
        if self.debug > 0:
            all_xyz_paths = all_xyz_paths[:self.debug]
        dlog.info(f"Loading from {total_xyz} geometry files")

        res = Parallel(n_jobs=n_workers)(
            delayed(read_qm9_xyz)(p) for p in all_xyz_paths
        )

        # Load in all of the data.
        for r in res:
            (qm9ID, smiles, canon, qm9properties, xyz, elements, zwitter) = r
            self.raw[qm9ID] = QM9DataPoint(
                qm9ID=qm9ID,
                smiles=(smiles, canon),
                qm9properties=qm9properties,
                xyz=xyz,
                elements=elements,
                zwitter=zwitter
            )

        dlog.info(f"Total number of raw QM9 data points: {len(self.raw)}")

    @time_func(dlog)
    def load_qm8_electronic_properties(self, path=None):
        """Function for loading Electronic properties for QM8 files.

        Parameters
        ----------
        path : str, optional
            Absolute path to the file containing the spectral information
            in the QM8 database. If None, checks for the QM8_EP_DATA_PATH
            environment variable.
        """

        if path is None:
            path = check_for_environment_variable(defaults.QM8_EP_ENV_VAR)

        dlog.info(f"Reading QM8 electronic properties from {path}")

        cc = 0
        all_props = []
        with open(path, 'r') as file:
            line = '#'
            while '#' in line:
                line = file.readline()
            while line != '':
                qm8_id, props = \
                    parse_QM8_electronic_properties(line.split())

                # There are cases where the QM8 ID will not be found in the
                # QM9 database. For example, when we skip all atoms (min
                # heavy atoms = 2).
                try:
                    self.raw[qm8_id].qm8properties = props
                except KeyError:
                    line = file.readline()
                    continue

                all_props.append(props)
                line = file.readline()
                cc += 1

        dlog.info(f"Total number of data points read from qm8: {cc}")

    @time_func(dlog)
    def load_oxygen_xanes(self, path=None):
        """Loads in the Oxygen XANES data from a pickle file of the following
        format.

        xanes = {
            qm9ID_1: {
                site_A : [spectra_A],
                site_B : [spectra_B],
                ...
            },
            ...
        }
        """

        if path is None:
            path = check_for_environment_variable(
                defaults.QM9_OXYGEN_FEFF_ENV_VAR
            )

        self.oxygenXANES_grid = np.linspace(526.98, 562.23, 80)
        # Note for Nitrogen it's [396.41, 431.06, 90]

        xanes = pickle.load(open(path, 'rb'))

        dlog.info(
            f"Loaded {len(xanes)} molecules of XANES successfully from {path}"
        )

        qm9IDs_to_use = intersection(list(xanes.keys()), list(self.raw.keys()))
        dlog.info(f"Length of the intersection is {len(qm9IDs_to_use)}")

        # We need to average the contributions for each site, which are listed
        # as dictionaries in XANES
        for qm9ID in qm9IDs_to_use:
            try:
                spectra = np.array([
                    spectrum for spectrum in xanes[qm9ID].values()
                ])
                self.raw[qm9ID].oxygenXANES = spectra.mean(axis=0)

            # If xanes[qm9ID] is None
            except AttributeError:
                continue


class QM9GraphDataset(torch.utils.data.Dataset, _SimpleLoadSaveOperations):
    """A special dataset which processes the exiting DataSet object into rdkit
    Chem mol and DGL graph objects. Also has analysis methods built on it
    for quick insights into the data, and individual objects.

    Attributes
    ----------
    dsname : str
        The name of the dataset, should match that of the QM9Dataset for
        consistency.
    raw : dict, optional
        A dictionary of the raw data as provided directly by QM9Dataset.raw.
    to_mol_called, to_graph_called, analyze_called : bool
        Whether or not the corresponding methods have been called.
    ml_data : list
        A list of the data as prepared to be processed by a collating function.
        Essentially data that is ready for pytorch to take over.

    Example
    -------
    # First initialize the standard datset
    qm9_dat = QM9Dataset(dsname='my_dataset', debug=1000)
    qm9_dat.load(...)
    ...
    qm9_dat_graph = QM9GraphDataset(qm9_dat)
    qm9_dat_graph.to_mol()
    qm9_dat_graph.analyze()
    qm9_dat_graph.to_graph()
    qm9_dat_graph.init_ml_data(scale_targets=...)
    """

    def __init__(self, ds=None, seed=None):
        self.dt_created = datetime.datetime.now()
        if ds is None:
            self.dsname = None
            self.raw = None
            self.oxygenXANES_grid = None
            self.nitrogenXANES_grid = None
        else:
            self.dsname = ds.dsname
            self.raw = ds.raw
            self.oxygenXANES_grid = ds.oxygenXANES_grid
            self.nitrogenXANES_grid = ds.nitrogenXANES_grid
        self.to_mol_called = False
        self.to_graph_called = False
        self.analyze_called = False
        self.ml_data = None
        self.target_metadata = None
        self.tvt_splits = None
        self.node_edge_features = None
        self.targets_to_use = None
        self.n_targets = None
        if seed is not None:
            dlog.info(f"Dataset seed set to {seed}")
        else:
            dlog.warning(f"Dataset seed set to {seed}")
        self.seed = seed

    def __getitem__(self, ii):
        return self.ml_data[ii]

    def __len__(self):
        return len(self.raw)

    def save_state(self, directory=None, override=False):
        """Dumps the class as a dictionary into a pickle file. It will save
        the object to the dsname directory as mld.pkl. Specifically, it saves
        to directory/dsname/mld.pkl."""

        self._save_state(
            class_type='mld', directory=directory, override=override
        )

    def load_state(self, dsname, directory=None):
        """Reloads the dataset of the specified name and directory."""

        self._load_state(class_type='mld', dsname=dsname, directory=directory)

    @staticmethod
    def _err_if_called(force):
        if force:
            dlog.warning(
                "You already called this but force is True: re-running"
            )
            return True
        else:
            dlog.error(
                "You already called this and force is False: "
                "exiting without re-running"
            )
            return False

    @time_func(dlog)
    def to_mol(self, canonical=False, n_workers=cpu_count(), force=False):
        """Fills the mol attribute in every DataPoint in raw.

        Properties
        ----------
        canonical : bool
            If True, uses the canonical SMILES code for the mol-generation,
            else uses the standard SMILES code.
        n_workers : int
            The number of workers to use during the generation of the mol
            objects, which can be time-consuming. Defaults to the number of
            available CPU's on your machine.
        force : bool
            If True and to_mol_called is True, will rerun the computation
            anyway.
        """

        if self.to_mol_called:
            go = QM9GraphDataset._err_if_called(force)
            if not go:
                return

        def _to_mol(ii, smiles):
            if canonical:
                mol = Chem.MolFromSmiles(smiles[1])
            else:
                mol = Chem.MolFromSmiles(smiles[0])
            return ii, mol, MolWt(mol)

        res = Parallel(n_jobs=n_workers)(
            delayed(_to_mol)(ii, dat.smiles) for ii, dat in self.raw.items()
        )

        for (qm9ID, mol, mw) in res:
            self.raw[qm9ID].mol = mol
            self.raw[qm9ID].mw = mw

        self.to_mol_called = True

    @time_func(dlog)
    def analyze(self, n_workers=cpu_count(), force=False):
        """Runs an in-depth analysis on every molecule in the dataset, using
        the analysis module from dgllife and a few in-house-developed analysis
        steps."""

        if self.analyze_called:
            go = QM9GraphDataset._err_if_called(force)
            if not go:
                return

        def _summarize(ii, mol):
            in_house = all_analysis(mol)
            return ii, summarize_a_mol(mol), in_house

        res = Parallel(n_jobs=n_workers)(
            delayed(_summarize)(ii, dat.mol) for ii, dat in self.raw.items()
        )

        for (qm9ID, summary, in_house_summary) in res:
            self.raw[qm9ID].summary = {**summary, **in_house_summary}

        self.analyze_called = True

    @time_func(dlog)
    def to_graph(
        self, node_method='weave', edge_method='canonical',
        n_workers=cpu_count(), force=False
    ):
        """Constructs the graphs for the entire raw attribute using the
        designated node and edge methods. Requires the to_mol method to have
        been called (in other words, requires that each QM9DataPoint has it's
        mol object initialized).

        Parameters
        ----------
        node_method : {'weave'}
        edge_method : {'canonical'}
        """

        if not self.to_mol_called:
            error = "You must call to_mol before calling this method - exiting"
            dlog.error(error)
            return

        if self.to_graph_called:
            go = QM9GraphDataset._err_if_called(force)
            if not go:
                return

        errors = []

        if node_method == 'weave':
            from dgllife.utils.featurizers import WeaveAtomFeaturizer
            fn = WeaveAtomFeaturizer(
                atom_data_field='features',
                atom_types=['H', 'C', 'N', 'O', 'F']
            )
            node_features = fn.feat_size()
        else:
            errors.append(f"Unknown node_method {node_method}")

        if edge_method == 'canonical':
            from dgllife.utils.featurizers import CanonicalBondFeaturizer
            fe = CanonicalBondFeaturizer(bond_data_field='features')
            edge_features = fe.feat_size()
        else:
            errors.append(f"Unknown edge_method {node_method}")

        if len(errors) > 0:
            for err in errors:
                dlog.error(err)
            dlog.error("Exiting without doing anything")
            return

        def _to_graph(ii, mol):
            return ii, mol_to_bigraph(
                mol, node_featurizer=fn, edge_featurizer=fe
            )

        res = Parallel(n_jobs=n_workers)(
            delayed(_to_graph)(ii, dat.mol) for ii, dat in self.raw.items()
        )

        for (qm9ID, graph) in res:
            self.raw[qm9ID].graph = graph

        self.to_graph_called = True
        self.node_edge_features = (node_features, edge_features)

    @staticmethod
    def collating_function_graph_to_vector(batch):
        """Collates the graph-fixed length vector combination. Recall that
        in this case, each element of batch is a three vector containing the
        graph, the target and the ID."""

        _ids = torch.tensor([xx[2] for xx in batch]).long()

        # Each target is the same length, so we can use standard batching for
        # it.
        targets = torch.tensor([xx[1] for xx in batch])

        # However, graphs are not of the same "length" (diagonally on the
        # adjacency matrix), so we need to be careful. Usually, dgl's batch
        # method would work just fine here, but for multi-gpu training, we
        # need to catch some subtleties, since the batch itself is split apart
        # equally onto many GPU's, but torch doesn't know how to properly split
        # a batch of graphs. So, we manually partition the graphs here, and
        # will batch the output of the collating function before training.
        # This is now just a list of graphs.
        graphs = [xx[0] for xx in batch]

        return (graphs, targets, _ids)

    @time_func(dlog)
    def init_ml_data(
        self, target_type='qm9properties', targets_to_use=[10],
        n_workers=cpu_count(), scale_targets=False, force=False
    ):
        """Initializes the ml_data attribute by pairing graphs with potential
        targets. The ml_data attribute contains three entries: [feature,
        target, id/metadata].

        Parameters
        ----------
        target_type : {'qm9properties', 'qm8properties', 'oxygenXANES'}
            The type of target. Some of these will require the user to have
            loaded in the properties beforehand, else they will not exist.
            Note that this must match the attribute in the QM9DataPoint class.
        targets_to_use : list, optional
            A list of integers specifying the specific targets to use. For
            example, having read in all the qm9 properties, if targets_to_use
            is [10], this will correspond to using only the 10th target in
            that list, which corresponds to U0.
        """

        if self.ml_data is not None:
            dlog.error("ml_data already initialized and force is False")
            go = QM9GraphDataset._err_if_called(force)
            if not go:
                return

        np.random.seed(self.seed)
        random.seed(self.seed)

        dlog.info(f"Using target type {target_type}")
        dlog.info(f"Using target indexes {targets_to_use}")
        dlog.info(f"Scaling targets: {scale_targets}")

        def _to_ml_data(datum):
            a = getattr(datum, target_type)
            if a is None:
                return None
            graph = datum.graph
            if targets_to_use is not None:
                a = [a[ii] for ii in targets_to_use]
            qm9ID = datum.qm9ID
            return [graph, a, qm9ID]

        res = Parallel(n_jobs=n_workers)(
            delayed(_to_ml_data)(datum) for datum in list(self.raw.values())
        )

        self.targets_to_use = targets_to_use
        self.ml_data = []
        for r in res:
            if r is None:
                continue
            (graph, target, qm9ID) = r
            self.ml_data.append([graph, target, qm9ID])

        dlog.info(f"Total number of ML-ready datapoints {len(self.ml_data)}")

        self.n_targets = len(self.ml_data[0][1])

        random.shuffle(self.ml_data)

        trgs = [xx[1] for xx in self.ml_data]
        mu, sd = mean_and_std(trgs)

        if not scale_targets:
            return

        self.target_metadata = [mu, sd]
        self.ml_data = [
            [
                xx[0],
                [(xx[1][vv] - mu[vv]) / sd[vv] for vv in range(len(xx[1]))],
                xx[2]
            ] for xx in self.ml_data
        ]

        dlog.info(f"Target metadata is {self.target_metadata}")
        trgs = [xx[1] for xx in self.ml_data]
        mean_and_std(trgs)

    def init_splits(
        self, p_tvt=(0.1, 0.1, None), method='random', force=False
    ):
        """Chooses the splits based on molecule criteria. The default is a
        random split. Note that the first time this method is called, the
        attribute tvt_splits will be set, but it will not allow the user to
        rewrite that split unless force=True. This is a failsafe mechanism to
        prevent bias in the data by constantly reshuffling the splits while
        evaluating the results."""

        if self.tvt_splits is not None:
            dlog.error("tvt_splits already initialized and force is False")
            go = QM9GraphDataset._err_if_called(force)
            if not go:
                return

        np.random.seed(self.seed)

        if method == 'random':
            s = RandomSampler(len(self.ml_data))
            s.shuffle_(self.seed)
            assert s.indexes_modified
            self.tvt_splits = s.split(p_tvt[0], p_tvt[1], p_train=p_tvt[2])
        else:
            critical = f"Invalid split method {method}"
            dlog.critical(critical)
            raise NotImplementedError(critical)

    def get_loaders(self, batch_sizes=(32, 32, 32)):
        """Returns the loaders as computed by the prior sampling."""

        # Initialize the subset objects
        testSubset = torch.utils.data.Subset(self, self.tvt_splits['test'])
        validSubset = torch.utils.data.Subset(self, self.tvt_splits['valid'])
        trainSubset = torch.utils.data.Subset(self, self.tvt_splits['train'])

        # Initialize the loader objects
        testLoader = torch.utils.data.DataLoader(
            testSubset, batch_size=batch_sizes[0], shuffle=False,
            collate_fn=QM9GraphDataset.collating_function_graph_to_vector
        )
        validLoader = torch.utils.data.DataLoader(
            validSubset, batch_size=batch_sizes[1], shuffle=False,
            collate_fn=QM9GraphDataset.collating_function_graph_to_vector
        )
        trainLoader = torch.utils.data.DataLoader(
            trainSubset, batch_size=batch_sizes[2], shuffle=True,
            collate_fn=QM9GraphDataset.collating_function_graph_to_vector
        )

        return {
            'test': testLoader,
            'valid': validLoader,
            'train': trainLoader
        }
