#!/usr/bin/env python3

import shutil

from crescendo.utils.managers.vec2vec_manager import Vec2VecManager
from crescendo.utils.arg_parser import global_parser


class TestVec2VecPipeline:

    def test_init_ml(self):

        args = global_parser(
            'vec2vec -d test123_2 --cache __test_2 init '
            '--path data/df_testing_data '
            '--scale-features --scale-targets'.split()
        )
        manager = Vec2VecManager(dsname=args.dsname, cache=args.cache)
        manager.init_ml(args)

    def test_prime(self):

        args = global_parser(
            'vec2vec -d test123_2 --cache __test_2 prime '
            '--ml-config configs/vec2vec_ml_config_template.yaml '
            '--slurm-config configs/slurm_config_template.yaml'.split()
        )
        manager = Vec2VecManager(dsname=args.dsname, cache=args.cache)
        manager.prime(config_path=args.ml_config, max_hp=args.max_hp)
        manager.write_SLURM_script(slurm_config=args.slurm_config)
        shutil.rmtree('__test_2')


class TestVec2VecPipelineDownsample:

    def test_init_ml(self):

        args = global_parser(
            'vec2vec -d test123_2 --cache __test_2 init '
            '--path data/df_testing_data '
            '--scale-features --scale-targets --downsample-train 20'.split()
        )
        manager = Vec2VecManager(dsname=args.dsname, cache=args.cache)
        manager.init_ml(args)

    def test_prime(self):

        args = global_parser(
            'vec2vec -d test123_2 --cache __test_2 prime '
            '--ml-config configs/vec2vec_ml_config_template.yaml '
            '--slurm-config configs/slurm_config_template.yaml'.split()
        )
        manager = Vec2VecManager(dsname=args.dsname, cache=args.cache)
        manager.prime(config_path=args.ml_config, max_hp=args.max_hp)
        manager.write_SLURM_script(slurm_config=args.slurm_config)
        shutil.rmtree('__test_2')
