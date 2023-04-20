import argparse
from argparse import HelpFormatter, ArgumentDefaultsHelpFormatter
from operator import attrgetter


# https://stackoverflow.com/questions/
# 12268602/sort-argparse-help-alphabetically
class SortingHelpFormatter(ArgumentDefaultsHelpFormatter, HelpFormatter):
    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)
        # self._max_help_position = 40
        # self._width = 100


def global_parser(sys_argv):
    ap = argparse.ArgumentParser(formatter_class=SortingHelpFormatter)

    ap.add_argument(
        '--smoke', dest='smoke', default=False, action="store_true",
        help='Smoke test on the California housing dataset'
    )

    # Initialize qm9 subparser
    # subparsers = ap.add_subparsers(
    #     help='Global options', dest='run_type'
    # )
    # smoke_parser = subparsers.add_parser(
    #     "smoke", formatter_class=SortingHelpFormatter,
    #     description='QM9 molecular data. The general protocol here is loading '
    #     'SMILES and target property data from QM9 .xyz files, processing the '
    #     'SMILES data into graph-representations, and using a message passing '
    #     'neural network to make predictions on fixed length vectors. The '
    #     'workflow is raw -> graph -> prime -> train -> eval.'
    # )
    # matrix_subparser = subparsers.add_parser("--matrix")
    # add_qm9_args(qm9_subparser)

    return ap.parse_args(sys_argv)
