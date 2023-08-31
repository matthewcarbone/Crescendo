import dgl
import random

from ase import Atom
import numpy as np
from pathlib import Path
from scipy.interpolate import InterpolatedUnivariateSpline
import torch
from tqdm import tqdm

from pymatgen.core.structure import Structure


type_encoding = {}
specie_am = []
for Z in range(1, 119):
    specie = Atom(Z)
    type_encoding[specie.symbol] = Z - 1
    specie_am.append(specie.mass)

type_onehot = torch.eye(len(type_encoding))
am_onehot = torch.diag(torch.tensor(specie_am))


def gdf(d, rmin=0.8, rmax=8.0, step_divisor=8.0):
    """Summary

    Parameters
    ----------
    d : TYPE
        Description
    rmin : float, optional
        Description
    rmax : float, optional
        Description
    step_divisor : float, optional
        Description

    Returns
    -------
    TYPE
        Description
    """

    step = (rmax - rmin) / step_divisor
    filter_ = np.arange(rmin, rmax + step, step)
    var = step
    return np.exp(-((d - filter_) ** 2) / var**2)


def construct_single_graph(
    structure,
    site,
    rmax=8.0,
    use_gauss_bl=False,
    node_featurization_method="onehot+mw",
):
    """Summary

    Parameters
    ----------
    structure : TYPE
        Description
    site : TYPE
        Description
    rmax : float, optional
        Description
    use_gauss_bl : bool, optional
        Description
    node_featurization_method : str, optional
        Description

    No Longer Returned
    ------------------
    TYPE
        Description

    No Longer Raises
    ----------------
    RuntimeError
        Description
    """

    num_nodes = len(structure)
    bond_lengths = []
    edge_a = []
    edge_b = []

    # handle the edges
    for ii in range(num_nodes):
        nearest_neighbors = structure.get_all_neighbors(
            r=rmax,
            include_index=True,
            include_image=True,
            sites=[structure[ii]],
        )[0]
        if len(nearest_neighbors) == 0:
            raise RuntimeError("len(nearest_neighbors)==0!")
        for jj in nearest_neighbors:
            edge_a.append(ii)
            edge_b.append(jj.index)
            distance = jj.distance_from_point(structure[ii].coords)
            if use_gauss_bl:
                bond_lengths.append(gdf(distance))
            else:
                bond_lengths.append(distance)
    bond_lengths = np.array(bond_lengths)
    if not use_gauss_bl:
        bond_lengths = bond_lengths.reshape(-1, 1)

    # node: one-hot element encoding
    if "onehot" in node_featurization_method:
        element_one_hot = [
            am_onehot[site.specie.number - 1].numpy() for site in structure
        ]
        if node_featurization_method == "onehot+mw":
            pass
        elif node_featurization_method == "onehot":
            element_one_hot = (np.array(element_one_hot) >= 1).astype(int)
        else:
            raise ValueError(
                "Unknown node_featurization_method "
                f"{node_featurization_method}"
            )
    elif node_featurization_method == "mw":
        element_one_hot = [Atom(site.specie.symbol).mass for site in structure]
        element_one_hot = torch.Tensor(element_one_hot).reshape(-1, 1)
    else:
        raise ValueError(
            f"Unknown node_featurization_method {node_featurization_method}"
        )
    element_one_hot = torch.Tensor(element_one_hot)

    # Indicate which node corresponds to the absorbing atom
    site_idx_lst = [0 for _ in range(num_nodes)]
    site_idx_lst[site] = 1
    site_idx = torch.Tensor(site_idx_lst).reshape(-1, 1)

    # Concatenate the features together
    node_features = torch.hstack((element_one_hot, site_idx))
    node_features = torch.Tensor(node_features).view(num_nodes, -1)

    edge_features = torch.Tensor(bond_lengths)
    edge_a = torch.LongTensor(edge_a)
    edge_b = torch.LongTensor(edge_b)
    graph = dgl.graph(data=(edge_a, edge_b), num_nodes=num_nodes)
    graph.ndata["features"] = node_features
    graph.edata["features"] = edge_features
    return graph


def construct_feff_dataset(root, shuffle_materials=True, seed=123, **kwargs):
    """Creates the ML dataset and saves it to disk. This function assumes
    that files are in a certain structure. Specifically, the root directory
    has a list of materials. In that directory, is a single POSCAR, and another
    directory called FEFF-XANES, in which there are directories corresponding
    to the absorbing sites. Within those are the results of the FEFF
    calculations.

    Parameters
    ----------
    root : os.PathLike
    shuffle_materials : bool, optional
        If True, shuffles the list of materials before iterating through them.
    **kwargs
        Keyword arguments passed to construct_single_graph.

    Returns
    -------
    list, list, list
        Lists of the graphs and spectra (and their origins, "names")
    """

    graphs = []
    spectra = []
    names = []

    materials = list(Path(root).iterdir())
    if shuffle_materials:
        random.seed(seed)
        random.shuffle(materials)

    for material in tqdm(materials):
        if not Path(material).is_dir():
            continue
        structure = Structure.from_file(Path(material) / "POSCAR")
        for site_directory in (Path(material) / "FEFF-XANES").iterdir():
            site = int(str(site_directory.stem).split("_")[0])
            graph = construct_single_graph(structure, site, **kwargs)
            try:
                feff_spectrum = np.loadtxt(Path(site_directory) / "xmu.dat")
            except FileNotFoundError:
                continue
            graphs.append(graph)
            spectra.append(feff_spectrum)
            names.append(f"{material.name}_{site}")

    # N x L x 6
    spectra = np.array(spectra)
    spectra = np.array([spectra[:, :, 0], spectra[:, :, 3]]).T.swapaxes(0, 1)

    return graphs, spectra, names


def interpolate_spectra(spectra, min_max=None, N=200):
    """Interpolates the provided spectra onto a common grid. The left and right
    bounds are either provided or are set to the maximum of the minimum lower
    bound and the minimum of the maximum upper bound.

    Parameters
    ----------
    spectra : np.ndarray
        Of shape N x L x 2, where N is the number of spectra, L is the length
        of the spectra, and the last two columns are the x and y axes of the
        spectra, respectively.
    min_max : list, optional
        If provided, this is a list/tuple of length 2 representing the
        user-provided minimum and maximum for the x-axis (energy range) of the
        spectra.
    N : int, optional
        The total number of points on the new grid.

    Returns
    -------
    dict
        The spectra of shape N x L, as well as the grid itself, in a
        dictionary.
    """

    if min_max is not None:
        _min = min_max[0]
        _max = min_max[1]
    else:
        _min = spectra[:, 0, 0].max()
        _max = spectra[:, -1, 0].min()

    grid = np.linspace(_min, _max, N)

    new_spectra = []
    for s in spectra:
        ius = InterpolatedUnivariateSpline(s[:, 0], s[:, 1], k=3)
        new_spectra.append(ius(grid))

    return {"spectra": np.array(new_spectra), "grid": grid}
