from pathlib import Path


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
