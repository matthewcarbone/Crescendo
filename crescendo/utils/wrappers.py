from functools import wraps
import warnings

from rich.console import Console

CONSOLE = Console()


def log_warnings(header_message):

    def inner(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings(record=True) as warnings_caught:
                res = func(*args, **kwargs)

                if warnings_caught:
                    CONSOLE.log(header_message, style="bold yellow")
                    for w in warnings_caught:
                        CONSOLE.log(w)

                return res

        return wrapper

    return inner
