#!/usr/bin/env python3

import time


def time_func(logging_stream, arg1=None):
    """Adapted from:
    http://scottlobdell.me/2015/04/decorators-arguments-python/"""

    def real_decorator(function):

        def wrapper(*args, **kwargs):

            aa = arg1
            if aa is None:
                aa = function.__name__

            t1 = time.time()
            x = function(*args, **kwargs)
            t2 = time.time()
            elapsed = t2 - t1
            logging_stream.info(f"{aa} done {elapsed:.02f} s")
            return x

        return wrapper

    return real_decorator
