import random

import numpy as np

from .half_cheetah import HalfCheetahEnv


class HalfCheetahVelEnv(HalfCheetahEnv):
    """Half-cheetah environment with target velocity, as described in [1]. The
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand.py

    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each
    time step a reward composed of a control cost and a penalty equal to the
    difference between its current velocity and the target velocity. The tasks
    are generated by sampling the target velocities from the uniform
    distribution on [0, 2].

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for
        model-based control", 2012
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """

    def __init__(self, max_episode_steps=200, out_of_distribution=False):
        self._max_episode_steps = max_episode_steps
        self.task_dim = 1
        self.out_of_distribution = out_of_distribution
        self.set_task(self.sample_tasks(1)[0])
        super(HalfCheetahVelEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self.goal_velocity)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost,
                     task=self.get_task())
        return observation, reward, done, infos

    def set_task(self, task):
        self.goal_velocity = task

    def get_task(self):
        return self.goal_velocity

    def sample_tasks(self, n_tasks, test=False):
        if self.out_of_distribution:
            if test:
                return [random.uniform(0.0, 2.0) for _ in range(n_tasks)]
            else:
                return [random.uniform(2.0, 3.0) for _ in range(n_tasks)]
        return [random.uniform(0.0, 3.0) for _ in range(n_tasks)]

    def reset_task(self, task=None, test=False):
        if test:
            print('\n\nTESTING\n\n')
        if task is None:
            task = self.sample_tasks(1, test)[0]
        self.set_task(task)
        # self.reset()


class HalfCheetahRandVelOracleEnv(HalfCheetahVelEnv):

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
            [self.goal_velocity]
        ])
