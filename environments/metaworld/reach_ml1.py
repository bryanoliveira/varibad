import numpy as np
import gym
from random import randint
from metaworld.benchmarks import ML1

class ReachML1Env(gym.Env):

    def __init__(self, max_episode_steps=150,out_of_distribution=False, n_train_tasks=50, n_test_tasks=10, **kwargs):
        super(ReachML1Env, self).__init__()
        self.train_env = ML1.get_train_tasks('reach-v1', out_of_distribution=out_of_distribution)
        self.test_env = ML1.get_test_tasks('reach-v1', out_of_distribution=out_of_distribution)
        self.train_tasks = self.train_env.sample_tasks(n_train_tasks)
        self.test_tasks = self.test_env.sample_tasks(n_test_tasks)
        self.tasks = self.train_tasks + self.test_tasks
        self.env = self.train_env #this env will change depending on the idx
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.goal_space_origin = np.array([0, 0.85, 0.175])
        self.current_task_idx = 0
        self.episode_steps = 0
        self._max_episode_steps = max_episode_steps
        # self.get_tasks_goals()
        # self.reset_task()

    def step(self, action):
        self.episode_steps += 1
        obs, reward, done, info = self.env.step(action)
        if self.episode_steps >= self._max_episode_steps:
            done = True
        return obs, reward, done, info

    def reset(self):
        self.episode_steps = 0
        return self.env.reset()

    def seed(self, seed):
        self.train_env.seed(seed)
        self.test_env.seed(seed)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def set_task(self, idx):
        self.current_task_idx = idx
        self.env = self.train_env if idx < len(self.train_tasks) else self.test_env
        self.env.set_task(self.tasks[idx])
        self._goal = self.tasks[idx]['goal']

    def get_task(self):
        return self.tasks[self.current_task_idx]['goal'] # goal_pos

    def reset_task(self, task=None, test=False):
        # aparently this is called only without idx, so tasks are always scrambled
        # we have to set anything only at test time
        if task is None:
            if test:
                task = randint(len(self.train_tasks), len(self.tasks) - 1)
            else:
                task = randint(0, len(self.train_tasks) - 1)
        self.set_task(task)

    def render(self):
        self.env.render()
    
    def get_tasks_goals(self):
        for idx in range(len(self.tasks)):
            self.reset_task(idx)
            _, _, _, info = self.step(self.action_space.sample())
            self.tasks[idx]['goal_pos'] = info['goal']
