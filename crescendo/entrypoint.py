import sys

from crescendo.utils.arg_parser import global_parser


def entrypoint():
    args = sys.argv[1:]
    parsed = global_parser(args)

    if parsed.smoke:
        run_smoke_test()
        return
