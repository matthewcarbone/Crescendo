#!/usr/bin/env python3

import shutil

from crescendo.utils.managers.qm9_manager import QM9Manager
from crescendo.utils.arg_parser import global_parser


class TestQM9Pipeline:

    def test_init_raw(self):

        args = global_parser(
            'qm9 -d test123 --cache __test --qm9-path data/qm9_test_data '
            'raw --qm8-path data/qm8_test_data.txt --no-oxygen-xanes'.split()
        )
        manager = QM9Manager(dsname=args.dsname, cache=args.cache)
        manager.init_raw(args)

    def test_init_graph(self):

        args = global_parser(
            'qm9 -d test123 --cache __test graph --analyze --scale-targets '
            '--canonical'.split()
        )
        manager = QM9Manager(dsname=args.dsname, cache=args.cache)
        manager.init_graph(args)

    def test_prime(self):

        args = global_parser(
            'qm9 -d test123 --cache __test prime --ml-config '
            'configs/ml_config_template.yaml '
            '--slurm-config configs/slurm_config_template.yaml'.split()
        )
        manager = QM9Manager(dsname=args.dsname, cache=args.cache)
        manager.prime(config_path=args.ml_config, max_hp=args.max_hp)
        manager.write_SLURM_script(slurm_config=args.slurm_config)
        shutil.rmtree('__test')
