from functools import wraps
import warnings

from rich.console import Console

CONSOLE = Console()


IGNORE = [
    "is an instance of `nn.Module` and is already saved during checkpointing",
]


def log_warnings(header_message=None):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings(record=True) as warnings_caught:
                res = func(*args, **kwargs)

                if warnings_caught:
                    if header_message is not None:
                        CONSOLE.log(header_message, style="bold yellow")
                    for w in warnings_caught:
                        w = str(w)
                        if any([xx in w for xx in IGNORE]):
                            continue
                        CONSOLE.log(w)

                return res

        return wrapper

    return inner
