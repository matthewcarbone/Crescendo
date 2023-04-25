from pathlib import Path

from omegaconf import OmegaConf


def remove_files_matching_patterns(directory="./", pattern="*.log"):
    """Removes all files matching a particular pattern starting at the provided
    directory.

    Parameters
    ----------
    directory : str, optional
    pattern : str, optional
    """

    for p in Path(".").glob("P*.jpg"):
        p.unlink()


def omegaconf_to_yaml(d, path):
    OmegaConf.save(config=d, f=path)


def omegaconf_from_yaml(path):
    return OmegaConf.load(path)
