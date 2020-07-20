#!/usr/bin/env python3

import numpy as np

from crescendo.utils.logger import logger_default as dlog


class Sampler:
    """Base sampling class.

    Attributes
    ----------
    n_points : int
        The number of total points in the database.
    indexes : np.ndarray
        A numpy array containing an ordered list of integers 0, ..., n_points.
        This attribute is modified by the shuffling/sorting methods of the
        Sampler.
    indexes_modified : bool
        A flag that switches to True if the indexes attribute is modified in
        any way, meaning shuffled or re-ordered. If False, this will throw a
        warning if the user tries to sampel the T/V/T splits from indexes
        without actually shuffling or sorting it, since this would be akin
        to taking indexes 0, 1, ..., N_test - 1 for testing, N_test,
        N_test + 1, ..., ... etc for validation, ... and similarly for
        training, which is usually not the standard use case.
    """

    def __init__(self, n_points):
        """Initializer.

        Parameters
        ----------
        n_points : int
            The number of total points in the database.
        """

        self.n_points = n_points
        self.indexes = np.array([ii for ii in range(self.n_points)])
        self.indexes_modified = False

    def shuffle_(self, seed=None):
        """Shuffles the indexes attribute. Operates in place.

        Parameters
        ----------
        seed : int, optional
            Seed for the numpy random number generator. Default is None
            (no seed, meaning re-seeded every time).
        """

        np.random.seed(seed)
        np.random.shuffle(self.indexes)
        self.indexes_modified = True

    def split(self, p_test, p_valid, p_train=None):
        """Executes a simple split over a total of n_points and returns the
        lists corresponding to the training, validation and testing splits.
        Will always split the indexes attribute in order, meaning:

        self.indexes[:i1] is the test set
        self.indexes[i1:i2] is the validation set
        self.indexes[i2:] is the testing set

        The indexes attribute should be shuffled or sorted prior to calling
        this method.

        Parameters
        ----------
        p_test, p_valid : float
            Proportion of the data for use in the testing/validation sets.
        p_train : float, optional
            The proportion of the data to be used for training. Note that it
            is not required that p_train = 1 - p_valid - p_test. If
            p_train + p_valid + p_test != 1, then p_train represents the
            proportion of the total data used for training. For example,
            if p_test = 0.1 and p_valid = 0.2 and p_train = None, then
            p_train will be 0.7. However, if p_train = 0.6, then 10% of the
            data will be discarded. This is useful for downsampling the
            training set to evaluate the effects of less training data on the
            ML algorithms.

        Returns
        -------
        dict
            A dictionary with keys "test", "valid" and "train". The
            corresponding values are lists (tuples) containing the indexes of
            these splits.
        """

        if p_valid + p_test >= 1.0:
            raise RuntimeError(
                "Proportions of testing and validation sets must sum to "
                "less than 1.0."
            )

        if not self.indexes_modified:
            dlog.warning(
                "Sampling from ordered indexes, in order! Your testing set "
                "will look like [0, 1, 2, ..., N_test - 1]. Read the "
                "documentation of the `Sampler` class for details."
            )

        if p_valid + p_test > 0.5:
            dlog.warning(
                "Your validation and testing sets sum to over 50 percent of "
                "the total data. Except under special circumstances, this is "
                "not recommended."
            )

        i1 = int(self.n_points * p_test)
        i2 = i1 + int(self.n_points * p_valid)

        if p_train is None:
            p_train = 1.0 - p_valid - p_test
        elif p_test + p_valid + p_train < 1.0:
            dlog.info("Training set is explicitly downsampled.")
        i3 = i2 + int(self.n_points * p_train)

        if p_test + p_valid + p_train > 1.0:
            raise RuntimeError(
                "Proportions of testing, validation and training sets must "
                "sum to 1.0 or less."
            )

        dlog.info(f"T/V/T props: {p_test}/{p_valid}/{p_train}")

        testing_set = self.indexes[:i1]
        validation_set = self.indexes[i1:i2]
        training_set = self.indexes[i2:i3]

        dlog.info(
            f"T/V/T lengths: {len(testing_set)}/{len(validation_set)}"
            f"/{len(training_set)}"
        )

        return {
            'test': tuple(testing_set),
            'valid': tuple(validation_set),
            'train': tuple(training_set)
        }
