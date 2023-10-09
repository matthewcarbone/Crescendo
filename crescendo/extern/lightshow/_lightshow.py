"""Code designed to work with the output of the Lightshow Python package.
Specifically, this module deals with materials data."""

from pathlib import Path
import random

import numpy as np
from PyAstronomy.pyasl import broadGaussFast
from pymatgen.core.structure import Structure
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def _load_structures(root, purge_structures=None):
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

    if purge_structures is None:
        purge_structures = []

    materials = {}
    for d in tqdm(list(Path(root).iterdir())):
        if not d.is_dir():
            continue
        if str(d.name) in purge_structures:
            print(f"Purging {d.name}")
            continue
        try:
            structure = Structure.from_file(d / "POSCAR")
        except FileNotFoundError:
            continue

        materials[d.name] = structure
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
                if spectra_type not in ["FEFF-XANES", "FEFF-EXAFS"]:
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


def _load_vasp_data(root, element, purge_structures=None):
    """Finds all mu2.txt files and loads them into numpy arrays. Also loads
    the structures since we need them for further processing."""

    if purge_structures is None:
        purge_structures = []

    spectra = {}
    structures = {}
    # metadata = {}
    errors = []
    for d in tqdm(list(Path(root).iterdir())):
        if not d.is_dir():
            continue
        if str(d.name) in purge_structures:
            print(f"Purging {d.name}")
            continue
        for site_directory in sorted(list((Path(d) / "VASP").iterdir())):
            if "SCF" in str(site_directory):
                continue
            poscar = Structure.from_file(site_directory / "POSCAR")
            site = int(str(site_directory.stem).split("_")[0])
            try:
                spectrum = np.loadtxt(Path(site_directory) / "mu2.txt")
            except FileNotFoundError:
                errors.append(str(site_directory))
                continue

            # If loading any of these failed, it's likely an indicator that
            # the calculation did not converge.
            scf = site_directory.parent / "SCF" / "scfenergy.txt"
            try:
                ecorehole = np.loadtxt(
                    Path(site_directory) / "ecorehole.txt"
                ).item()
                efermi = np.loadtxt(Path(site_directory) / "efermi.txt").item()
                e_scf = np.loadtxt(scf).item()
            except (ValueError, FileNotFoundError):
                errors.append(str(site_directory))
                continue
            name = f"{d.name}_{site}"

            # prim = poscar.get_primitive_structure()
            # total_element_type = np.sum([xx.specie.symbol == element for
            # xx in prim])
            # normalization = 1.0 / prim.volume

            # Prendergast shift
            # delta_SCF = ECH - SCF
            # delta = delta_SCF - Efermi
            # EV = EV + delta
            delta_SCF = ecorehole - e_scf
            delta = delta_SCF - efermi
            spectrum[:, 0] = spectrum[:, 0] + delta
            # spectrum[:, 1] = spectrum[:, 1] / normalization
            spectra[name] = spectrum
            structures[name] = poscar
            # metadata[name] = {
            #     "ecorehole": ecorehole,
            #     "efermi": efermi,
            #     "escf": e_scf
            # }
    return spectra, structures, errors


def _interpolate_spectra(spectra, interpolation_grid, broadening=0.59):
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

        if broadening is not None:
            if broadening > 0.0:
                interpolated_sp = broadGaussFast(
                    interpolation_grid, interpolated_sp, broadening
                )

        interpolated_spectra[key] = interpolated_sp
    return interpolated_spectra


def _double_check(structures, spectra_keys, element):
    for key in spectra_keys:
        mpid, index = key.split("_")
        assert structures[mpid][int(index)].specie.symbol == element, key


def _prepare_feff_arrays(structures, spectra_interp, featurizer):
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


def _prepare_vasp_arrays(structures, spectra_interp, featurizer):
    """Unlike the feff data, the structures are keyed by materialid and site.

    Parameters
    ----------
    structures : TYPE
        Description
    spectra_interp : TYPE
        Description
    featurizer : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """

    returned_names = []
    returned_features = []
    returned_spectra = []

    for key, I in tqdm(spectra_interp.items()):
        material = structures[key]

        # This is wasteful but I'd prefer to be careful
        features = featurizer(material)

        # Always index 0 for VASP supercell, since the absorber is always
        # at the top!
        returned_features.append(features[0, :])
        returned_spectra.append(I)
        returned_names.append(key)

    return (
        returned_names,
        np.array(returned_features),
        np.array(returned_spectra),
    )


def _prepare_feff_dataset(
    path,
    grid,
    featurizer,
    element=None,
    spectra_type="FEFF-XANES",
    purge_structures=None,
):
    if purge_structures is None:
        purge_structures = []
    print("Loading structures...")
    structures = _load_structures(path, purge_structures)
    print(f"Loading {spectra_type} spectra...")
    if spectra_type in ["FEFF-XANES", "FEFF-EXAFS"]:
        spectra, spectra_errors = _load_feff_spectra(path, spectra_type)
    else:
        raise ValueError(f"Unknown type of spectrum {spectra_type}")
    print("Interpolating spectra...")
    spec_interp = _interpolate_spectra(spectra, grid)
    if element is not None:
        print("Double checking indexes...")
        _double_check(structures, spec_interp.keys(), element)
    print("Converting to arrays...")
    names, feats, spectra_final = _prepare_feff_arrays(
        structures, spec_interp, featurizer
    )
    print("Done")
    return {
        "names": names,
        "node_features": feats,
        "spectra": spectra_final,
        "spectra_errors": spectra_errors,
    }


def _prepare_vasp_dataset(
    path, grid, featurizer, element, purge_structures, broadening
):
    print("Loading structures and spectra...")
    spectra, structures, spectra_errors = _load_vasp_data(
        path, element, purge_structures
    )
    print("Interpolating spectra...")
    spec_interp = _interpolate_spectra(spectra, grid, broadening)
    if element is not None:
        print("Double checking indexes...")
        for structure in structures.values():
            assert structure[0].specie.symbol == element
    print("Converting to arrays...")
    names, feats, spectra_final = _prepare_vasp_arrays(
        structures, spec_interp, featurizer
    )
    print("Done")
    return {
        "names": names,
        "node_features": feats,
        "spectra": spectra_final,
        "spectra_errors": spectra_errors,
    }


def prepare_dataset(
    path,
    grid,
    featurizer,
    element=None,
    spectra_type="FEFF-XANES",
    purge_structures=None,
    broadening=None,
):
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
    spectra_type : str, optional

    Returns
    -------
    dict
        The names, node features and spectra.
    """

    if purge_structures is None:
        purge_structures = []

    if "FEFF" in spectra_type:
        return _prepare_feff_dataset(
            path, grid, featurizer, element, spectra_type, purge_structures
        )
    elif "VASP" == spectra_type:
        return _prepare_vasp_dataset(
            path, grid, featurizer, element, purge_structures, broadening
        )
    else:
        raise ValueError(f"Unknown spectra type {spectra_type}")


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
