#!/usr/bin/env python3


from rdkit import Chem

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


def has_n_membered_ring(mol, n=None) -> bool:
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
    return mol.HasSubstructMatch(Chem.MolFromSmarts(f'[r{n}]'))


def is_aromatic(mol) -> bool:
    """If the molecule has any aromatic pattern, returns True, else
    returns False. This is checked by trying to locate the substructure
    match for the string '[a]'.

    Returns
    -------
    bool
    """

    return mol.HasSubstructMatch(aromatic_pattern)


def has_double_bond(mol) -> bool:
    """Checks the molecule for double bonds. Note that by default this
    method assumes the data point is from QM9, and only checks the
    atoms capable of forming double bonds in QM9, so, it will only check
    C, N and O, and their 6 combinations.

    Returns
    -------
    bool
    """

    return any([mol.HasSubstructMatch(p) for p in double_bond_patterns])


def has_triple_bond(mol) -> bool:
    """Checks the molecule for triple bonds. Note that by default this
    method assumes the data point is from QM9, and only checks the
    atoms capable of forming triple bonds in QM9, so, it will only check
    C#C, C#N and N#N.

    Returns
    -------
    bool
    """

    return any([mol.HasSubstructMatch(p) for p in triple_bond_patterns])


def has_hetero_bond(mol) -> bool:
    """Checks the molecule for heter bonds. Note that by default this
    method assumes the data point is from QM9, and only checks the
    atoms capable of forming hetero bonds in QM9, so, it will only check
    C~O, C~N and N~O.

    Returns
    -------
    bool
    """

    return any([mol.HasSubstructMatch(p) for p in hetero_bond_patterns])
