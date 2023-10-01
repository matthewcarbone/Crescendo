from .logger import logger  # noqa

__version__ = 'dev'  # semantic-version-placeholder

# This works, but we're going with the rich.console for now

# import logging

# from rich.logging import RichHandler

# logger = logging.getLogger("rich")
# shell_handler = RichHandler()
# logger.addHandler(shell_handler)
# logger.setLevel(11)
# logger.propagate = False

# in the modules
# log = logging.getLogger("rich")
