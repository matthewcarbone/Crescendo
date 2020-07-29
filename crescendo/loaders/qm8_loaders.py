#!/usr/bin/env python3

"""Module for loading in Data from QM8 database."""

def parse_QM8_electronic_properties(
    props,
    selected_properties=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
):
    """Parses a list of strings into the correct floats that correspond to the
    electronic properties in the QM8 database.

    the properties are as follows (1-indexed):
        1  : index
        2  : RI-CC2/def2TZVP E1 in atomic units
        3  : RI-CC2/def2TZVP E2 in atomic units
        4  : RI-CC2/def2TZVP f1 in atomic units in length representation
        5  : RI-CC2/def2TZVP f2 in atomic units in length representation
        6  : LR-TDPBE0/def2SVP E1 in atomic units
        7  : LR-TDPBE0/def2SVP E2 in atomic units
        8  : LR-TDPBE0/def2SVP f1 in atomic units in length representation
        9  : LR-TDPBE0/def2SVP f2 in atomic units in length representation
        10 : LR-TDPBE0/def2TZVP E1 in atomic units
        11 : LR-TDPBE0/def2TZVP E2 in atomic units
        12 : LR-TDPBE0/def2TZVP f1 in atomic units in length representation
        13 : LR-TDPBE0/def2TZVP f2 in atomic units in length representation
        14 : LR-TDCAM-B3LYP/def2TZVP E1 in atomic units
        15 : LR-TDCAM-B3LYP/def2TZVP E2 in atomic units
        16 : LR-TDCAM-B3LYP/def2TZVP f1 in atomic units in length representation
        17 : LR-TDCAM-B3LYP/def2TZVP f2 in atomic units  in length representation
            
    The relevant ones (2 through 17 inclusive) will be returned in a new list
    with each element being of the correct type.

    Parameters
    ----------
    props : list[str]
        Initial properties in string format.
    selected_properties : List[int]

    Returns
    -------
    int, list[float]
        The QM8 ID and list of properties (list[float]).
    """

    qm8_id = int(props[0])
    other = props[1:]
    other = [
        float(prop) for ii, prop in enumerate(other)
        if ii in selected_properties
    ]
    return (qm8_id, other)

def read_qm8(qm8_path):
    """Function for reading Electronic Spectra for QM8 files.

    Parameters
    ----------
    qm8_path : str
        Absolute path to the  file.

    Returns
    -------
    int, list[float]
        The QM8 ID and list of properties (list[float]).
    """

    with open(qm8_path, 'r') as file:
        line = '#'
        while '#' in line:
            line = file.readline()
        qm8_id, props = \
            parse_QM8_electronic_properties(line.split())

    return (qm8_id, props)

