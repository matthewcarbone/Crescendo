from dunamai import Version

version = Version.from_any_vcs()
__version__ = version.serialize()
del version


# __version__ = "0.0.1"


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
