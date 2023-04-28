from contextlib import contextmanager
import json
from pathlib import Path
from subprocess import Popen, PIPE
from time import perf_counter

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
