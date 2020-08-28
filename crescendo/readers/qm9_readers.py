#!/usr/bin/env python3

"""Utilities for reading data corresponding to the QM9 dataset."""

import numpy as np


def parse_QM8_electronic_properties(props, selected_properties=None):
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

    Note that the properties we'll generally be using are [0, 13, 14, 15, 16]

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
    if selected_properties is None:
        other = [float(prop) for ii, prop in enumerate(other)]
    else:
        other = [
            float(prop) for ii, prop in enumerate(other)
            if ii in selected_properties
        ]
    return (qm8_id, other)


def parse_QM9_scalar_properties(props, selected_properties=None):
    """Parses a list of strings into the correct floats that correspond to the
    molecular properties in the QM9 database.

    Only the following properties turn out to be statistically relevant in this
    dataset: selected_properties=[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 14]. These
    properties are the statistically independent contributions as calculated
    via a linear correlation model, and together the capture >99% of the
    variance of the dataset.

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
    selected_properties : List[int], optional
        Selected properties. If None, take all the properties.

    Returns
    -------
    int, list[float]
        The QM9 ID and list of properties (list[float]).
    """

    qm9_id = int(props[1])
    other = props[2:]

    if selected_properties is None:
        other = [float(prop) for ii, prop in enumerate(other)]
    else:
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
