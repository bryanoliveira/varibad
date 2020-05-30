"""
Main scripts to start experiments.
Takes a flag --env-type (see below for choices) and loads the parameters used in the paper.
"""
import argparse
import glob
import os
import warnings

import torch

# setup devices and args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0) # int for validation
parser.add_argument('--no-gpu', action='store_true')
parser.add_argument('--env-type', default='gridworld_varibad')
args, rest_args = parser.parse_known_args()
os.environ['ENABLE_GPU'] = "0" if args.no_gpu else "1"
os.environ["GPU_ID"] = str(args.gpu)


# get configs
from config.metaworld import args_metaworld_reach_varibad
from config.gridworld import \
    args_grid_oracle, args_grid_belief_oracle, args_grid_rl2, args_grid_varibad
from config.mujoco import \
    args_mujoco_cheetah_dir_oracle, args_mujoco_cheetah_dir_rl2, args_mujoco_cheetah_dir_varibad, \
    args_mujoco_cheetah_vel_oracle, args_mujoco_cheetah_vel_rl2, args_mujoco_cheetah_vel_varibad, \
    args_mujoco_ant_dir_oracle, args_mujoco_ant_dir_rl2, args_mujoco_ant_dir_varibad, \
    args_mujoco_walker_oracle, args_mujoco_walker_rl2, args_mujoco_walker_varibad
from learner import Learner
from metalearner import MetaLearner


def main(args):
    env = args.env_type

    # --- MetaWorld ---
    if env == 'ml1_reach_varibad':
        args = args_metaworld_reach_varibad.get_args(rest_args)

    # --- GridWorld ---

    # standard
    if env == 'gridworld_oracle':
        args = args_grid_oracle.get_args(rest_args)
    elif env == 'gridworld_belief_oracle':
        args = args_grid_belief_oracle.get_args(rest_args)
    elif env == 'gridworld_varibad':
        args = args_grid_varibad.get_args(rest_args)
    elif env == 'gridworld_rl2':
        args = args_grid_rl2.get_args(rest_args)

    # --- MUJOCO ---

    # - AntDir -
    elif env == 'mujoco_ant_dir_oracle':
        args = args_mujoco_ant_dir_oracle.get_args(rest_args)
    elif env == 'mujoco_ant_dir_rl2':
        args = args_mujoco_ant_dir_rl2.get_args(rest_args)
    elif env == 'mujoco_ant_dir_varibad':
        args = args_mujoco_ant_dir_varibad.get_args(rest_args)
    #
    # - CheetahDir -
    elif env == 'mujoco_cheetah_dir_oracle':
        args = args_mujoco_cheetah_dir_oracle.get_args(rest_args)
    elif env == 'mujoco_cheetah_dir_rl2':
        args = args_mujoco_cheetah_dir_rl2.get_args(rest_args)
    elif env == 'mujoco_cheetah_dir_varibad':
        args = args_mujoco_cheetah_dir_varibad.get_args(rest_args)
    #
    # - CheetahVel -
    elif env == 'mujoco_cheetah_vel_oracle':
        args = args_mujoco_cheetah_vel_oracle.get_args(rest_args)
    elif env == 'mujoco_cheetah_vel_rl2':
        args = args_mujoco_cheetah_vel_rl2.get_args(rest_args)
    elif env == 'mujoco_cheetah_vel_varibad':
        args = args_mujoco_cheetah_vel_varibad.get_args(rest_args)
    #
    # - Walker -
    elif env == 'mujoco_walker_oracle':
        args = args_mujoco_walker_oracle.get_args(rest_args)
    elif env == 'mujoco_walker_rl2':
        args = args_mujoco_walker_rl2.get_args(rest_args)
    elif env == 'mujoco_walker_varibad':
        args = args_mujoco_walker_varibad.get_args(rest_args)

    # make sure we have log directories for mujoco
    if 'mujoco' in env:
        try:
            os.makedirs(args.agent_log_dir)
        except OSError:
            files = glob.glob(os.path.join(args.agent_log_dir, '*.monitor.csv'))
            for f in files:
                os.remove(f)
        eval_log_dir = args.agent_log_dir + "_eval"
        try:
            os.makedirs(eval_log_dir)
        except OSError:
            files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
            for f in files:
                os.remove(f)

    # warning
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, run it with num_processes=1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < env._max_episode_steps.')

    # start training
    if args.disable_varibad:
        # When the flag `disable_varibad` is activated, the file `learner.py` will be used instead of `metalearner.py`.
        # This is a stripped down version without encoder, decoder, stochastic latent variables, etc.
        learner = Learner(args)
    else:
        learner = MetaLearner(args)
    learner.train()


if __name__ == '__main__':
    main(args)
