#!/usr/bin/env python3

import inspect

from crescendo.utils.logger import logger_default

# Import submodules for testing
from tests.test_datasets import TestBaseLoader, TestCSVLoader, TestQMXLoader
from tests.test_samplers import TestSampler


def run_all_methods(obj):
    attrs = (getattr(obj, name) for name in dir(obj))
    methods = filter(inspect.ismethod, attrs)
    for method in methods:
        method()


if __name__ == '__main__':

    logger_default.disabled = True
    run_all_methods(TestBaseLoader())
    run_all_methods(TestCSVLoader())
    run_all_methods(TestQMXLoader())
    run_all_methods(TestSampler())
    logger_default.disabled = False
