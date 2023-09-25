"""Code designed to work with the output of the Lightshow Python package.
Specifically, this module deals with materials data."""

from pathlib import Path
import random

import numpy as np
from pymatgen.core.structure import Structure
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def _load_structures(root):
    """Finds all POSCAR files and loads them into Pymatgen Structure objects.
    Labels these structures by its parent directory's name. Assumes that the
    path's last piece (the filename) is the index for the material.

    Parameters
    ----------
    root : os.PathLike
        The location to search one layer down.

    Returns
    -------
    dict
        A dictionary keyed by the material identifier, and the values are the
        pymatgen.core.structure.Structure objects.
    """

    materials = {}
    for d in tqdm(list(Path(root).iterdir())):
        if not d.is_dir():
            continue
        materials[d.name] = Structure.from_file(d / "POSCAR")
    return materials


def _load_feff_spectra(root, spectra_type="FEFF-XANES"):
    """Finds all xmu.dat files and loads them into numpy arrays."""

    spectra = {}
    errors = []
    for d in tqdm(list(Path(root).iterdir())):
        if not d.is_dir():
            continue
        for site_directory in sorted(list((Path(d) / spectra_type).iterdir())):
            site = int(str(site_directory.stem).split("_")[0])
            try:
                if "FEFF" in spectra_type:
                    feff_spectrum = np.loadtxt(
                        Path(site_directory) / "xmu.dat"
                    )
                else:
                    raise NotImplementedError(
                        f"Unknown spectra_type={spectra_type}"
                    )
            except FileNotFoundError:
                errors.append(str(site_directory))
                continue
            name = f"{d.name}_{site}"
            spectra[name] = np.array(
                [feff_spectrum[:, 0], feff_spectrum[:, 3]]
            ).T
    return spectra, errors


def _interpolate_spectra(spectra, interpolation_grid):
    """Interpolates the provided spectra onto a common grid. The left and right
    bounds are either provided or are set to the maximum of the minimum lower
    bound and the minimum of the maximum upper bound.

    Parameters
    ----------
    spectra : np.ndarray
        Of shape N x L x 2, where N is the number of spectra, L is the length
        of the spectra, and the last two columns are the x and y axes of the
        spectra, respectively.
    interpolation_grid : np.ndarray
        The new grid to interpolate onto.

    Returns
    -------
    dict
        The newly interpolated spectra.
    """

    interpolated_spectra = {}
    for key, s in spectra.items():
        ius = InterpolatedUnivariateSpline(s[:, 0], s[:, 1], k=3)
        interpolated_sp = ius(interpolation_grid)
        interpolated_sp[interpolated_sp < 0.0] = 0.0
        interpolated_spectra[key] = interpolated_sp
    return interpolated_spectra


def _double_check(structures, spectra_keys, element):
    for key in spectra_keys:
        mpid, index = key.split("_")
        assert structures[mpid][int(index)].specie.symbol == element


def _prepare_dataset(structures, spectra_interp, featurizer):
    """Creates a dataset of materials and their spectra.

    Parameters
    ----------
    structures : TYPE
        Description
    spectra_interp : TYPE
        Description
    featurizer : callable
        Converts a material to node features.

    Returns
    -------
    TYPE
        Description
    """

    returned_names = []
    returned_features = []
    returned_spectra = []

    _materials_node_features = {}

    for key, I in tqdm(spectra_interp.items()):
        mpid, index = key.split("_")
        index = int(index)
        material = structures[mpid]

        features = _materials_node_features.get(mpid, None)
        if features is None:
            _materials_node_features[mpid] = featurizer(material)
            features = _materials_node_features[mpid]

        returned_features.append(features[index, :])
        returned_spectra.append(I)
        returned_names.append(key)

    return (
        returned_names,
        np.array(returned_features),
        np.array(returned_spectra),
    )


def prepare_dataset(path, grid, featurizer, element=None):
    """Summary

    Parameters
    ----------
    path : os.PathLike
        The path to where the data is.
    grid : array_like
        The new grid for the spectra to be interpolated onto.
    featurizer : callable
        Converts a material into a list of node features.
    element : str
        The element we double check against. If None, this check is skipped.

    Returns
    -------
    dict
        The names, node features and spectra.
    """

    print("Loading structures...")
    structures = _load_structures(path)
    print("Loading spectra...")
    spectra, spectra_errors = _load_feff_spectra(path)
    print("Interpolating spectra...")
    spec_interp = _interpolate_spectra(spectra, grid)
    if element is not None:
        print("Double checking indexes...")
        _double_check(structures, spec_interp.keys(), element)
    print("Converting to arrays...")
    names, feats, spectra_final = _prepare_dataset(
        structures, spec_interp, featurizer
    )
    print("Done")
    return {
        "names": names,
        "node_features": feats,
        "spectra": spectra_final,
        "spectra_errors": spectra_errors,
    }


def save_dataset(
    path, data, train_prop=0.8, split_type="random", random_state=42
):
    """Saves a dataset to disk as .npy files in a compatible fashion for
    Crescendo.

    Parameters
    ----------
    path : TYPE
        Description
    data : TYPE
        Description
    train_prop : float, optional
        Description
    split_type : str, optional
        Description
    """

    X = data["node_features"]
    Y = data["spectra"]
    names = data["names"]

    if split_type == "random":
        print("Random split...")

        (
            X_train,
            X_test,
            Y_train,
            Y_test,
            names_train,
            names_test,
        ) = train_test_split(
            X, Y, names, test_size=1.0 - train_prop, random_state=random_state
        )

        X_val, X_test, Y_val, Y_test, names_val, names_test = train_test_split(
            X_test,
            Y_test,
            names_test,
            test_size=0.5,
            random_state=random_state,
        )

    elif split_type == "material":
        print("Material split...")

        # Split the training and testing sets by material type
        unique_materials = list(set([n.split("_")[0] for n in names]))
        random.seed(random_state)
        random.shuffle(unique_materials)

        # Figure out how many of each split there eare
        L = len(unique_materials)
        N_train = int(train_prop * L)
        N_val = (L - N_train) // 2

        # Split the names
        names_train = unique_materials[:N_train]
        names_val = unique_materials[N_train : N_train + N_val]
        names_test = unique_materials[N_train + N_val :]
        assert set(names_train).isdisjoint(set(names_val))
        assert set(names_train).isdisjoint(set(names_test))
        assert set(names_test).isdisjoint(set(names_val))

        # Get the indexes
        ii_train = [
            ii
            for ii, name in enumerate(names)
            if name.split("_")[0] in names_train
        ]
        ii_val = [
            ii
            for ii, name in enumerate(names)
            if name.split("_")[0] in names_val
        ]
        ii_test = [
            ii
            for ii, name in enumerate(names)
            if name.split("_")[0] in names_test
        ]
        assert set(ii_train).isdisjoint(set(ii_val))
        assert set(ii_train).isdisjoint(set(ii_test))
        assert set(ii_test).isdisjoint(set(ii_val))

        # And construct the data themselves
        X_train = X[ii_train, :]
        X_val = X[ii_val, :]
        X_test = X[ii_test, :]
        Y_train = Y[ii_train, :]
        Y_val = Y[ii_val, :]
        Y_test = Y[ii_test, :]
        names_train = [names[ii] for ii in ii_train]
        names_val = [names[ii] for ii in ii_val]
        names_test = [names[ii] for ii in ii_test]

    else:
        raise ValueError(f"Unknown split type {split_type}")

    print(f"Train    X={X_train.shape}, Y={Y_train.shape}")
    print(f"Val      X={X_val.shape}, Y={Y_val.shape}")
    print(f"Test     X={X_test.shape}, Y={Y_test.shape}")

    # Write the files to disk
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)

    np.save(path / "X_train.npy", X_train)
    np.save(path / "X_val.npy", X_val)
    np.save(path / "X_test.npy", X_test)

    np.save(path / "Y_train.npy", Y_train)
    np.save(path / "Y_val.npy", Y_val)
    np.save(path / "Y_test.npy", Y_test)

    with open(path / "names_train.txt", "w") as f:
        for line in names_train:
            f.write(f"{line}\n")

    with open(path / "names_val.txt", "w") as f:
        for line in names_val:
            f.write(f"{line}\n")

    with open(path / "names_test.txt", "w") as f:
        for line in names_test:
            f.write(f"{line}\n")
