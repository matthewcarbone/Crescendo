#!/usr/bin/env python3

"""Module for loading in data from the QM9 database."""

import os as os
import pickle as pickle
import glob2
from ntpath import basename
import numpy as np
from rdkit import Chem

from typing import List

try:
    # Used for to_pmg_molecule method
    import pymatgen.core.structure as pmgstruc
    _pmg_present = True
except ImportError:
    _pmg_present = False

from crescendo.utils.logger import logger_default as dlog
from crescendo.datasets.base import _BaseCore
from crescendo.utils.timing import time_func
from crescendo.featurizers.graphs import mol_to_graph_via_DGL, \
    get_number_of_classes_per_feature


aromatic_pattern = Chem.MolFromSmarts('[a]')

double_bond_patterns = [
    Chem.MolFromSmarts('C=C'), Chem.MolFromSmarts('C=O'),
    Chem.MolFromSmarts('C=N'), Chem.MolFromSmarts('O=O'),
    Chem.MolFromSmarts('O=N'), Chem.MolFromSmarts('N=N')
]

triple_bond_patterns = [
    Chem.MolFromSmarts('C#C'), Chem.MolFromSmarts('C#N'),
    Chem.MolFromSmarts('N#N')
]

hetero_bond_patterns = [
    Chem.MolFromSmarts('C~O'), Chem.MolFromSmarts('C~N'),
    Chem.MolFromSmarts('N~O')
]


def check_for_qm9_environment_variable():
    """Checks the os.environ dictionary for the QM9_DATA_PATH environment
    variable. If it exists, returns the path, else raises a ValueError and
    logs a critical level error to the logger."""

    qm9_directory = os.environ.get("QM9_DATA_PATH", None)
    if qm9_directory is None:
        error_msg = \
            "No path specified for QM9 directory, either " \
            "as argument or environment variable $QM9_DATA_PATH."
        dlog.critical(error_msg)
        raise RuntimeError(error_msg)
    return qm9_directory


class QM9SmilesDatum:
    """Functions for determining structures from a smiles input using rdkit
    chem.

    Current Structure types
        .ring -contains any ring
        .ring5 -contains a 5 atom ring
        .ring4 -contains a 4 atom ring
        .aromatic -contains an aromatic structure
        .doublebond -contains a double bond with the combinations of carbon,
        oxygen, and nitrogen
        .triplebond -contains a triple bond with the combinations of carbon
        and nitrogen
        .singlebond -does not contain .doublebond .triplebond and .aromatic

    Example
    -------
    #Molecule Benzene aromatic structure
    >>> d = QM9SmilesDatum('C1=CC=CC=C1')
    >>> d.is_aromatic()
    True
    """

    def __init__(self, smiles, other_props, xyz, elements, zwitter, qm9_id):
        """
        Parameters
        ----------
        smiles : str
            smiles of target molecule as string
        """

        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.other_props = other_props
        self.xyz = xyz
        self.elements = elements
        self.zwitter = zwitter
        self.qm9_id = qm9_id

    def to_graph(
        self, atom_feature_list=['type', 'hybridization'],
        bond_feature_list=['type']
    ):
        """Initializes the graph attribute of the molecule object. See
        crescendo.featurizer.graphs.mol_to_graph_via_DGL for more details."""

        return mol_to_graph_via_DGL(
            self.mol, atom_feature_list, bond_feature_list
        )

    def has_n_membered_ring(self, n=None) -> bool:
        """Returns True if the mol attribute (the molecule object in rdkit
        representing the Smiles string) has an n-membered ring.

        Parameters
        ----------
        n : int, optional
            The size of the ring. If None, will simply check if the molecule
            has any ring (using the substructure matching string '[r]' instead
            of [f'r{n}']). Default is None.

        Returns
        -------
        bool
            True if a match is found. False otherwise.
        """

        n = '' if n is None else n
        return self.mol.HasSubstructMatch(Chem.MolFromSmarts(f'[r{n}]'))

    def is_aromatic(self) -> bool:
        """If the molecule has any aromatic pattern, returns True, else
        returns False. This is checked by trying to locate the substructure
        match for the string '[a]'.

        Returns
        -------
        bool
        """

        return self.mol.HasSubstructMatch(aromatic_pattern)

    def has_double_bond(self) -> bool:
        """Checks the molecule for double bonds. Note that by default this
        method assumes the data point is from QM9, and only checks the
        atoms capable of forming double bonds in QM9, so, it will only check
        C, N and O, and their 6 combinations.

        Returns
        -------
        bool
        """

        return any([
            self.mol.HasSubstructMatch(p)
            for p in double_bond_patterns
        ])

    def has_triple_bond(self) -> bool:
        """Checks the molecule for triple bonds. Note that by default this
        method assumes the data point is from QM9, and only checks the
        atoms capable of forming triple bonds in QM9, so, it will only check
        C#C, C#N and N#N.

        Returns
        -------
        bool
        """

        return any([
            self.mol.HasSubstructMatch(p)
            for p in triple_bond_patterns
        ])

    def has_hetero_bond(self) -> bool:
        """Checks the molecule for heter bonds. Note that by default this
        method assumes the data point is from QM9, and only checks the
        atoms capable of forming hetero bonds in QM9, so, it will only check
        C~O, C~N and N~O.

        Returns
        -------
        bool
        """

        return any([
            self.mol.HasSubstructMatch(p)
            for p in hetero_bond_patterns
        ])

    def to_pmg_molecule(self):
        """Convenience method which turns the current QM9 datum into a Pymatgen
        Molecule, which can be processed further in other useful ways due to
        that classes' built-in methods."""

        return pmgstruc.Molecule(species=self.elements, coords=self.xyz)

    def as_dict(self) -> dict:
        """Convenience method which turns the current QM9 datum into a
        dictionary formatted by the attributes of the object. Can be used for
        serialization or de-serialization in a JSON format."""

        return dict(vars(self))

    @staticmethod
    def from_dict(dictionary):
        return QM9SmilesDatum(**dictionary)


def parse_QM8_electronic_properties(
    props,
    selected_properties=[0, 13, 14, 15, 16]
):
    """Parses a list of strings into the correct floats that correspond to the
    electronic properties in the QM8 database.

    the properties are as follows (1-indexed):
        1  : index
        2  : RI-CC2/def2TZVP E1 in au
        3  : RI-CC2/def2TZVP E2 in au
        4  : RI-CC2/def2TZVP f1 in au in length representation
        5  : RI-CC2/def2TZVP f2 in au in length representation
        6  : LR-TDPBE0/def2SVP E1 in au
        7  : LR-TDPBE0/def2SVP E2 in au
        8  : LR-TDPBE0/def2SVP f1 in au in length representation
        9  : LR-TDPBE0/def2SVP f2 in au in length representation
        10 : LR-TDPBE0/def2TZVP E1 in au
        11 : LR-TDPBE0/def2TZVP E2 in au
        12 : LR-TDPBE0/def2TZVP f1 in au in length representation
        13 : LR-TDPBE0/def2TZVP f2 in au in length representation
        14 : LR-TDCAM-B3LYP/def2TZVP E1 in au
        15 : LR-TDCAM-B3LYP/def2TZVP E2 in au
        16 : LR-TDCAM-B3LYP/def2TZVP f1 in au in length representation
        17 : LR-TDCAM-B3LYP/def2TZVP f2 in au in length representation

    Note `au` = atomic units

    Parameters
    ----------
    props : list[str]
        Initial properties in string format.
    selected_properties : List[int]

    Returns
    -------
    int, list[float]
        The QM9 ID and list of properties (list[float]).
    """

    qm8_id = int(props[0])
    other = props[1:]
    other = [
        float(prop) for ii, prop in enumerate(other)
        if ii in selected_properties
    ]
    return (qm8_id, other)


def parse_QM9_scalar_properties(
    props,
    selected_properties=[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 14]
):
    """Parses a list of strings into the correct floats that correspond to the
    molecular properties in the QM9 database.

    Only the following properties turn out to be statistically relevant in this
    dataset: selected_properties=[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 14]

    Note, according to the paper (Table 3)
    https://www.nature.com/articles/sdata201422.pdf
    the properties are as follows (1-indexed):
        1  : gdb9 index (we'll ignore)
        2  : identifier
        3  : "A" (GHz) rotational constant
        4  : "B" (GHz) rotational constant
        5  : "C" (GHz) rotational constant
        6  : "mu" (Debeye) dipole moment
        7  : "alpha" (a0^3) isotropic polarizability
        8  : "HOMO energy" (Ha)
        9  : "LUMO energy" (Ha)
        10 : "E gap" (Ha) 8-9 (might be uHa?)
        11 : "<R^2>" (a0^2) electronic spatial extent
        12 : "zpve" (Ha) zero-point vibrational energy
        13 : "U0" (Ha) internal energy at 0 K
        14 : "U" (Ha) internal energy at 198.15 K
        15 : "H" (Ha) enthalpy at 298.15 K
        16 : "G" (Ha) gibbs free energy at 298.15 K
        17 : "Cv" (cal/molK) heat capacity at 298.15 K

    The relevant ones (2 through 17 inclusive) will be returned in a new list
    with each element being of the correct type.

    Parameters
    ----------
    props : list[str]
        Initial properties in string format.
    selected_properties : List[int]
        The statistically independent subset of properties needed to capture
        the majority (>99%) of the variance in the QM9 dataset.

    Returns
    -------
    int, list[float]
        The QM9 ID and list of properties (list[float]).
    """

    qm9_id = int(props[1])
    other = props[2:]
    other = [
        float(prop) for ii, prop in enumerate(other)
        if ii in selected_properties
    ]
    return (qm9_id, other)


def read_qm9_xyz(xyz_path, canonical=True):
    """Function for reading .xyz files like those present in QM9. Note this
    does not read in geometry information, just properties and SMILES codes.
    For a detailed description of the properties contained in the qm9 database,
    see this manuscript: https://www.nature.com/articles/sdata201422.pdf

    Parameters
    ----------
    xyz_path : str
        Absolute path to the xyz file.
    canonical : bool
        Whether or not to use the canonical SMILES codes instead of the
        standard ones.

    Returns
    -------
    Tuple containing relevant information and boolean flag for whether or not
    the molecule is a Zwitter-ionic compound or not.
    """

    with open(xyz_path, 'r') as file:
        n_atoms = int(file.readline())
        qm9_id, other_props = \
            parse_QM9_scalar_properties(file.readline().split())

        elements = []
        xyzs = []
        for ii in range(n_atoms):
            line = file.readline().replace('.*^', 'e').replace('*^', 'e')
            line = line.split()
            elements.append(str(line[0]))
            xyzs.append(np.array(line[1:4], dtype=float))

        xyzs = np.array(xyzs)

        # Skip extra vibrational information
        file.readline()

        # Now read the SMILES code
        smiles = file.readline().split()
        _smiles = smiles[int(canonical)]

        zwitter = '+' in smiles[0] or '-' in smiles[0]

    return (qm9_id, _smiles, other_props, xyzs, elements, zwitter)


def generate_qm9_pickle(
    qm9_directory: str = None, write_loc: str = './qm9_data.pickle',
    custom_range: List[int] = None
) -> List:
    """Given a path to the QM9 directory, creates and writes a .pickle file
    representing the entire QM9 database.

    Parameters
    ----------
    qm9_directory : str
        Location of QM9 database files. Can load from path.
    write_loc : str
        Where pickle file should be written.
    custom_range : list
        Subset of integers to selectively load in.

    Returns
    -------
    List of QM9 molecules formatted by the read_qm9_xyz function.
    """

    if qm9_directory is None:
        qm9_directory = check_for_qm9_environment_variable()

    entries = glob2.glob(qm9_directory + "/*.xyz")

    if custom_range is not None:
        prefix = 'dsgdb9nsd_'
        suffix = '.xyz'
        # Isolate the numbers of available QM9 values
        entry_numbers = {
            int(entry.split('_')[1].split('.')[0]) for entry in entries
        }
        use_numbers = entry_numbers.intersection(set(custom_range))
        to_use_entries = [
            prefix + str(entry).zfill(6) + suffix for entry in
            use_numbers
        ]

    else:
        to_use_entries = entries

    molecules = []
    for ent in to_use_entries:
        mol_path = os.path.join(qm9_directory, ent)
        molecules.append(read_qm9_xyz(mol_path))

    if write_loc:
        with open(write_loc, 'wb') as f:
            pickle.dump(molecules, f)

    return molecules


class QMXDataset(_BaseCore):
    """Container for the QMX data, where X is some integer. Although not the
    proper notation, we refer to X as in general, either 8 or 9 (usually),
    where X=max number of heavy atoms (C, N, O and F)/molecule."""

    def __init__(self, *args, debug=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw = dict()
        self.qm8_electronic_properties = None
        self.featurized = dict()
        self.debug = debug

    @property
    def max_heavy_atoms(self):
        return self._max_heavy_atoms

    @max_heavy_atoms.setter
    def max_heavy_atoms(self, a):
        assert isinstance(a, int)
        assert a > 0
        dlog.info(f"Max heavy atoms set to {a}")
        self._max_heavy_atoms = a

    @property
    def keep_zwitter(self):
        return self._keep_zwitter

    @keep_zwitter.setter
    def keep_zwitter(self, z):
        assert isinstance(z, bool)
        dlog.info(f"Keeping zwitterions set to {z}")
        self._keep_zwitter = z

    @property
    def canonical(self):
        return self._canonical

    @canonical.setter
    def canonical(self, c):
        assert isinstance(c, bool)
        dlog.info(f"Canonical SMILES set to {c}")
        self._canonical = c

    @time_func(dlog)
    def load(
        self,
        path=None,
        max_heavy_atoms=9,
        keep_zwitter=False,
        canonical=True,
        log_every=10000
    ):
        """Loads in the QM9 data as set via the path in the initializer, and
        also optionally other auxiliary data, such as spectra.

        Parameters
        ----------
        path : str
            Path to the directory containing the qm9 .xyz files. For instance,
            if your xyz files are in directory /Users/me/data, then that should
            be the path. If path is None by default, it will check the
            os.environ dictionary for QM9_DATA_PATH, and if that does not
            exist, it will throw an error.
        max_heavy_atoms : int
            Maximum number of total heavy atoms (C, N, O, F) allowed in the
            dataset. By default, QM9 allows for... wait for it... 9 heavy
            atoms, but we can reduce this to, e.g., 8 to match other subsets
            used in the literature. Default is 9.
        keep_zwitter : bool
            If True, will keep zwitterionic compounds
            (https://en.wikipedia.org/wiki/Zwitterion) in the database. Default
            is False.
        canonical : bool
            If True, will use the canonical SMILES codes. Default is True.
        """

        if path is None:
            path = check_for_qm9_environment_variable()
        dlog.info(f"Loading QM9 from {path}")

        self.max_heavy_atoms = max_heavy_atoms
        self.keep_zwitter = keep_zwitter
        self.canonical = canonical

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

            (qm9_id, smiles, other_props, xyzs, elements, zwitter) = \
                read_qm9_xyz(current_path, canonical=self.canonical)

            # No need to do any of this work if we're taking all of the
            # possible molecules in the QM9 dataset.
            if self.max_heavy_atoms < 9:
                n_heavy = sum([e != 'H' for e in elements])
                if n_heavy > self.max_heavy_atoms:
                    continue

            if not self.keep_zwitter and zwitter:
                continue

            self.raw[qm9_id] = QM9SmilesDatum(
                smiles, other_props, xyzs, elements, zwitter, qm9_id
            )

        dlog.info(f"Total number of data points: {len(self.raw)}")

    def load_qm8_electronic_properties(self, path):
        """Function for loading Electronic properties for QM8 files.

        Parameters
        ----------
        path : str
            Absolute path to the file containing the spectral information
            in the QM8 database.
        """

        self.qm8_electronic_properties = dict()

        with open(path, 'r') as file:
            line = '#'
            while '#' in line:
                line = file.readline()
            while line != '':
                qm8_id, props = \
                    parse_QM8_electronic_properties(line.split())
                self.qm8_electronic_properties[qm8_id] = props
                line = file.readline()

    def analyze(self, n=None):
        """Performs the analysis of the currently loaded QM9 Dataset.

        Parameters
        ----------
        n : int, optional
            The size of the rings it checking the data set for

        Returns
        -------
        A Dictionary containing the amount of molecules containing each
        structure contained in the loaded data
            Current Structures
                aramotic
                double bonds
                triple bonds
                hetero bonds
                n membered rings
        """

        analysis = {
            'num_aromatic': 0,
            'num_double_bond': 0,
            'num_triple_bond': 0,
            'num_hetero_bond': 0,
            'num_n_membered_ring': 0
            }

        for qmx_id in self.raw:
            if self.raw[qmx_id].is_aromatic():
                analysis['num_aromatic'] += 1
            if self.raw[qmx_id].has_double_bond():
                analysis['num_double_bond'] += 1
            if self.raw[qmx_id].has_triple_bond():
                analysis['num_triple_bond'] += 1
            if self.raw[qmx_id].has_hetero_bond():
                analysis['num_hetero_bond'] += 1
            if self.raw[qmx_id].has_n_membered_ring(n):
                analysis['num_n_membered_ring'] += 1

        return analysis

    def featurize(self, featurizer, **kwargs):
        """This method is the workhorse of the QMXLoader. It will featurize
        the raw data depending on the user settings."""

        if featurizer == 'to_graph':

            # Get the number of classes per feature, used in initializing
            # the MPNN later.
            atom_f_list = kwargs['atom_feature_list']
            bond_f_list = kwargs['bond_feature_list']
            n_class_per_feature = get_number_of_classes_per_feature(
                atom_f_list, bond_f_list
            )

            # Convert every object in self.raw -> graphs
            return {
                _id: datum.to_graph(atom_f_list, bond_f_list)
                for _id, datum in self.raw.items()
            }, n_class_per_feature

        else:
            critical = f"Unknown featurizer: {featurizer}"
            dlog.critical(critical)
            raise RuntimeError(critical)

    def write_file(self, filename: str = 'QMdb', fmt: str = 'pickle'):
        """Write dataset into serialized form for later access."""

        if fmt in ['pickle', 'pckl', 'pkl', 'binary']:
            if len(filename.split('.')) == 1:
                filename = f'{filename}.pkl'
            pickle.dump(
                self, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL
            )

        else:
            critical = f"Your specified format {fmt} is not supported."
            dlog.critical(critical)
            raise ValueError(critical)
