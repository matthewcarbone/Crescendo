#!/usr/bin/env python3

"""
Featurization module related to geometry.
"""

from typing import List, Tuple
import numpy as np
from crescendo.featurizers.base import Featurizer

# Atomic radii (rough spread) in Angstroms for select atoms.
_spread_map = {"H": 0.53, "C": 0.67, "N": 0.56, "O": 0.48, "F": 0.42, "S": 0.88}


class PdfFeaturizer:
    def featurize(
        xyz: np.ndarray,
        elements: List[str],
        sd: float = 0.05,
        grid_p: Tuple = (0, 10, 100),
    ) -> np.ndarray:
        """
        Computes the pair distribution function of a molecule. We do this
        by treating the grid as the discrete points. For instance, see A2 in
        https://royalsocietypublishing.org/doi/full/10.1098/rsta.2018.0413#d3e1147
        where the pair distribution function is defined
        as the scaled probability of finding two atoms a distance 'r' apart.
        Returns the probability as a function of distance that two atoms are
        found at that distance apart.
        Parameters
        ----------
        xyz : np.ndarray
            An N x d array (d is the dimension, usually 3) of the atom locations.
        elements : List[str]
            A list of strings consisting of the elements.
        sd : float
            Standard deviation of the normalized Gaussians used to approximate
            the delta function in the pdf calculation.
        """

        N = xyz.shape[0]
        assert N == len(elements)
        grid = np.linspace(*grid_p)
        ans = np.zeros(grid.shape)
        for ii in range(N):
            b_i = _spread_map[elements[ii]]
            for jj in range(ii + 1, N):
                b_j = _spread_map[elements[jj]]
                r_ij = np.sqrt(np.sum((xyz[ii, :] - xyz[jj, :]) ** 2))
                ans += np.exp(-((grid - r_ij) ** 2) / 2.0 / sd ** 2) * b_i * b_j
        ans[1:] /= grid[1:]
        dx = grid[1] - grid[0]

        return ans / np.sum(dx * ans)
