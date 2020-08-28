#!/usr/bin/env python3

"""Module for loading in data from the QM9 database."""

import datetime
from ntpath import basename
import os as os

import glob2
import pickle as pickle

from crescendo import defaults
from crescendo.utils.logger import logger_default as dlog
from crescendo.utils.py_utils import check_for_environment_variable
from crescendo.readers.qm9_readers import parse_QM8_electronic_properties, \
    read_qm9_xyz
from crescendo.utils.timing import time_func


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
        zwitter=None,
        qm8properties=None,
        oxygenXANES=None,
        nitrogenXANES=None
    ):
        self.qm9ID = qm9ID
        self.smiles = smiles
        self.qm9properties = qm9properties
        self.xyz = xyz
        self.elements = elements
        self.zwitter = zwitter
        self.nheavy = sum([e != 'H' for e in self.elements])
        self.mol = None
        self.mw = None
        # self.mol = Chem.MolFromSmiles(smiles)
        # self.mw = MolWt(self.mol)


class QM9Dataset:
    """Container for the QM9 data. This is meant to be a standalone dataset
    that is only truly dependent on internal (crescendo) packages, pure python
    and numpy. The next iteration of this dataset is the QM9MLDataset, which
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
        the object to the dsname directory as raw.pkl."""

        if directory is None:
            directory = check_for_environment_variable(defaults.QM9_DS_ENV_VAR)

        if os.path.isdir(directory) and not override:
            error = \
                f"Directory {directory} exists and override is False - " \
                "exiting and not overwriting"
            dlog.error(error)
            return

        elif os.path.isdir(directory) and override:
            warning = \
                f"Directory {directory} exists and override is True - " \
                "overwriting saved dataset"
            dlog.warning(warning)

        full_dir = f"{directory}/{self.dsname}"
        os.makedirs(full_dir, exist_ok=True)

        full_path = f"{full_dir}/raw.pkl"

        d = self.__dict__
        pickle.dump(d, open(full_path, 'wb'), protocol=defaults.P_PROTOCOL)

    def load_state(self, dsname, directory=None):
        """Reloads the dataset of the specified name and directory."""

        if directory is None:
            directory = check_for_environment_variable(defaults.QM9_DS_ENV_VAR)

        full_path = f"{directory}/{dsname}/raw.pkl"
        d = pickle.load(open(full_path, 'rb'))

        for key, value in d.items():
            setattr(self, key, value)

    @time_func(dlog)
    def load(self, path=None, log_every=10000):
        """Loads in **only** the QM9 raw data from .xyz files.

        Parameters
        ----------
        path : str, optional
            Path to the directory containing the qm9 .xyz files. For instance,
            if your xyz files are in directory /Users/me/data, then that should
            be the path. If path is None by default, it will check the
            os.environ dictionary for QM9_DATA_PATH, and if that does not
            exist, it will throw an error.
        log_every : int
            Each time we hit log_every iterations during loading, the output
            will be logged to the logger.
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

        # Load in all of the data.
        for ii, current_path in enumerate(all_xyz_paths):

            if ii % log_every == 0 and ii != 0:
                pc = ii / total_xyz * 100.0
                dlog.info(
                    f"latest read from: {basename(current_path)} ({pc:.00f}%)"
                )

            (qm9ID, smiles, canon, qm9properties, xyz, elements, zwitter) = \
                read_qm9_xyz(current_path)

            self.raw[qm9ID] = QM9DataPoint(
                qm9ID=qm9ID,
                smiles=(smiles, canon),
                qm9properties=qm9properties,
                xyz=xyz,
                elements=elements,
                zwitter=zwitter
            )

        dlog.info(f"Total number of raw QM9 data points: {len(self.raw)}")

    def load_qm8_electronic_properties(
        self, path=None, selected_properties=None
    ):
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
        with open(path, 'r') as file:
            line = '#'
            while '#' in line:
                line = file.readline()
            while line != '':
                qm8_id, props = \
                    parse_QM8_electronic_properties(
                        line.split(), selected_properties=selected_properties
                    )
                self.raw[qm8_id].qm8properties = props
                line = file.readline()
            cc += 1

        dlog.info(f"Total number of data points read from qm8: {cc}")
