#!/usr/bin/env python3

import inspect

from crescendo.utils.logger import logger_default

# Import submodules for testing
from tests.test_datasets import TestBaseLoaders, TestCSVLoader, TestQMXLoader
from tests.test_samplers import TestSampler
from tests.test_data_containers import TestBaseContainer, TestArrayContainer


def run_all_methods(obj):
    attrs = (getattr(obj, name) for name in dir(obj))
    methods = filter(inspect.ismethod, attrs)
    for method in methods:
        method()


if __name__ == '__main__':

    logger_default.disabled = True
    run_all_methods(TestBaseLoaders())
    run_all_methods(TestCSVLoader())
    run_all_methods(TestQMXLoader())
    run_all_methods(TestSampler())
    run_all_methods(TestBaseContainer())
    run_all_methods(TestArrayContainer())
    logger_default.disabled = False
