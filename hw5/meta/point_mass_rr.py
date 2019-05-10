import numpy as np 
from gym import spaces 
from gym import Env 

class PointEnv(Env):
    """
    point mass on a 2-d plane
    goals sampled randomly from a square
    """

    def __init__(self, num_tasks=1):
        self.reset_task()
        self.reset()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, is_evaluation=False):
        """
        sample a new task randomly

        problem 3:
        make training and evaluation goals disjoint sets
        if 'is_evaluation' is true, sample from the evaluatoin set,
        otherwise sample from the training set
        """
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)

        # checkerboard pattern for evaluation vs. training
        # odd -> training, even -> test
        if is_evaluation:
            if not np.abs(x) % 2 > 1:
                if x < 0:
                    x -= 1
                else:
                    x += 1
            if not np.abs(y) % 2 > 1:
                if y < 0:
                    y -= 1
                else:
                    y += 1
        else:
            if not np.abs(x) % 2 < 1:
                if x < 0:
                    x += 1
                else:
                    x -= 1
            if not np.abs(y) % 2 < 1:
                if y < 0:
                    y += 1
                else:
                    y -= 1
        self._goal = np.array([x, y])

    def reset(self):
        self._state = np.array([0, 0], dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        return np.copy(self._state)

    def reward_function(self, x, y):
        return - (x ** 2 + y ** 2) ** 0.5

    def step(self, action):
        x, y = self._state
        # compute reward, add penalty for large actions instead of clipping them
        x -= self._goal[0]
        y -= self._goal[1]
        # check if task is complete
        done = abs(x) < .01 and abs(y) < .01
        reward = self.reward_function(x, y)
        # move to next state
        self._state = self._state + action
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

    def seed(self, seed):
        np.random.seed = seed