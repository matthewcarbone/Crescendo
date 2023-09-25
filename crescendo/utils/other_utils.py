from contextlib import contextmanager
from filelock import FileLock
import json
from pathlib import Path
from subprocess import Popen, PIPE
from time import perf_counter
import yaml

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


@contextmanager
def Timer():
    start = perf_counter()
    yield lambda: perf_counter() - start


def run_command(cmd):
    """Execute the external command and get its exitcode, stdout and
    stderr."""

    with Timer() as dt:
        proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
        out, err = proc.communicate()
        exitcode = proc.returncode

    return {
        "exitcode": exitcode,
        "output": out.decode("utf-8").strip(),
        "error": err.decode("utf-8").strip(),
        "elapsed": dt,
    }


def save_json(d, path):
    with open(path, "w") as outfile:
        json.dump(d, outfile, indent=4, sort_keys=True)


def read_json(path):
    with open(path, "r") as infile:
        dat = json.load(infile)
    return dat


def save_yaml(d, path):
    yaml.dump(d, open(path, "w"))


def read_yaml(path):
    return yaml.safe_load(open(path, "r"))


class GlobalCache:
    """
    Parameters
    ----------
    d : os.PathLike
        Path to (likely a temp) directory containing the cache.
    cache_name : str, optional
        Name of the cache. Don't change this.
    """

    def __init__(self, d, cache_name=".crescendo_cache.yaml"):
        self._path = Path(d) / cache_name

    def read(self):
        """Loads a cache yaml file which can be shared between different calls
        of train from a provided directory.

        Returns
        -------
        dict
        """

        if not Path(self._path).exists():
            return {}

        with FileLock(f"{self._path}.lock"):
            d = read_yaml(self._path)

        return d

    def save(self, d):
        with FileLock(f"{self._path}.lock"):
            save_yaml(d, self._path)
