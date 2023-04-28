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


def run_command(cmd):
    """Execute the external command and get its exitcode, stdout and
    stderr."""

    t0 = perf_counter()
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    out, err = proc.communicate()
    exitcode = proc.returncode
    dt = perf_counter() - t0

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
