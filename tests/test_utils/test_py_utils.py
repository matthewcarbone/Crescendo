#!/usr/bin/env python3

import os
import pytest

from crescendo.utils.py_utils import intersection, flatten_list, \
    check_for_environment_variable


class TestPyUtilsBasic:

    def test_intersection(self):
        l1 = [1, 2, 3, 4, 8, 10, 20]
        l2 = [2, 17, 10, 4, 20, 19, 100, 1002]
        l3 = intersection(l1, l2)
        assert set(l3) == set([2, 4, 10, 20])

    def test_flatten_list(self):
        l1 = [[1, 2, 3], [4, 8, 10, 20], [123]]
        l2 = flatten_list(l1)
        assert l2 == [1, 2, 3, 4, 8, 10, 20, 123]

    def test_check_for_environment_variable(self):
        var = "definitely_not_an_env_variable"
        if var not in os.environ.keys():
            with pytest.raises(Exception):
                _ = check_for_environment_variable(var)

        var = '___FAKE'
        fake_path = '___fake_path'
        os.environ[var] = fake_path
        assert check_for_environment_variable(var) == fake_path
