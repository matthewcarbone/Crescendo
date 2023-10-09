"""M3GNet for Materials XAS!"""

from pathlib import Path

import numpy as np
import torch

from crescendo.extern.m3gnet._featurizer import (
    featurize_material,
    _load_default_featurizer,
)


N_GRID = 200

GRIDS = {
    "FEFF": {
        "Ti": np.linspace(4965, 5075, N_GRID),
        "Cu": np.linspace(8983, 9124, N_GRID),
    },
    "VASP": {"Ti": np.linspace(4715, 4765, N_GRID)},
}

ALLOWED_XAS_TYPES = ["XANES"]

ALLOWED_THEORY = ["FEFF"]

ALLOWED_EDGES = ["K"]

ALLOWED_ABSORBERS = ["Ti"]

ZOO_PATH = Path(__file__).parent.resolve() / "zoo"


def predict_spectrum(structure, model, featurizer, absorber):
    """Predicts the XAS of a structure given a model and absorbing atom.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
    model
        Should be a machine learning model that has a predict method defined
        on it.
    absorber : str
        The absorbing atom species.
    featurizer : callable
        Should return the features of the structure.

    Returns
    -------
    np.ndarray
    """

    features = featurize_material(structure, model=featurizer)
    indexes = [
        ii
        for ii, site in enumerate(structure)
        if site.specie.symbol == absorber
    ]
    features = features[indexes, :]
    return model.predict(features).mean(axis=0)


def get_predictor(
    xas_type="XANES",
    theory="FEFF",
    edge="K",
    absorber="Ti",
    version="v230925-oxides-materials_split",
):
    """Returns a dictionary of the default configuration for predicting the
    XANES of materials.

    Parameters
    ----------
    xas_type : str, optional
        Either XANES or EXAFS.
    theory : str, optional
        The level of theory at which the calculation was performed.
    edge : str, optional
        Description
    absorber : str, optional
        The atom doing the absorbing.
    version : str, optional
        The version of the model. If there is a wildcard "*" in the version,
        it will interpret this as an ensemble and will attempt a match for all
        models of that type.

    Returns
    -------
    callable
        A function which takes a pymatgen.core.structure.Structure as input and
        returns a dictionary, resolved by site, of the predictions.
    """

    if xas_type not in ALLOWED_XAS_TYPES:
        raise NotImplementedError("Choose from xas_type in ['XANES'] only")

    if theory not in ALLOWED_THEORY:
        raise NotImplementedError("Only FEFF theory available right now")

    if absorber not in ALLOWED_ABSORBERS:
        raise NotImplementedError(
            f"Choose from absorber from {ALLOWED_ABSORBERS}"
        )

    if edge not in ALLOWED_EDGES:
        raise NotImplementedError(f"Choose edge from {ALLOWED_EDGES}")

    # Currently this is all that's implemented
    def featurizer(structure):
        return featurize_material(structure, model=_load_default_featurizer())

    # Model signatures will be very specific
    directory = ZOO_PATH / xas_type / theory / f"{edge}-edge" / absorber
    model_signature = f"{version}.pt"

    # Ensemble
    if "*" in version:
        model_paths = directory.glob(model_signature)
        models = [torch.load(s) for s in model_paths]

    # Not an ensemble
    else:
        model_signature = directory / model_signature
        models = [torch.load(model_signature)]

    def predictor(structure, site_resolved=True):
        features = featurizer(structure)
        indexes = [
            ii
            for ii, site in enumerate(structure)
            if site.specie.symbol == absorber
        ]
        features = torch.tensor(features[indexes, :])
        preds = np.array(
            [m(features).detach().numpy() for m in models]
        ).swapaxes(0, 1)

        if site_resolved:
            return {site: pred.squeeze() for site, pred in zip(indexes, preds)}
        else:
            mu = preds.mean(axis=1)  # (sites, nw)
            sd = preds.std(axis=1)
            true_mean = mu.mean(axis=0)
            true_std = np.sqrt(np.sum(sd**2, axis=0))
            return true_mean, true_std

    return predictor