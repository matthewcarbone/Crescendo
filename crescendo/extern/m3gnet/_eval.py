from crescendo.extern.m3gnet._featurizer import (
    featurize_material,
    _load_default_featurizer,
)


def predict_spectrum(
    structure, model, absorber, featurizer=_load_default_featurizer()
):
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
