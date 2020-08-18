#!/usr/bin/env python3

from crescendo.utils.py_utils import intersection


class TestPyUtilsBasic:

    def test_intersection(self):
        l1 = [1, 2, 3, 4, 8, 10, 20]
        l2 = [2, 17, 10, 4, 20, 19, 100, 1002]
        l3 = intersection(l1, l2)
        assert set(l3) == set([2, 4, 10, 20])
