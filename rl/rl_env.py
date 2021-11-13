"""
train env
"""

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
import torch
import pandas as pd


def str_to_embbeding(series, tokenizer,model, max_len=None):
    # tokenize
    tokenized = series.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    # 2  padding
    if max_len is None:
        max_len=0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    # Masking
    attention_mask = np.where(padded != 0, 1, 0)

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    #predict
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:,0,:].numpy()

    return  pd.DataFrame(features).to_numpy().tolist()


class MainEnv(gym.Env):
    """
    the main goal is
    """

    def __init__(self,
                 data,
                 action_dim=5
                 ):

        self.data_dict = data
        gb = data.groupby('reward')
        data_dict = {x: gb.get_group(x) for x in gb.groups}
        self.data_dict = data_dict

        self.seed()
        self.steps_beyond_done = None
        self.current_session = None
        self.np_random = None
        self._iter = 0

        self.action_space = spaces.Discrete(
            action_dim
            )
        self.levels = list(data_dict.keys())
        self.iter_len = len(self.levels)

        self._observation_space = None
        self.state_cols = None
        self.current_level = None
        # high = self.train_features[self.state_cols].max(axis=0).values
        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.current_reward = 0

    @property
    def observation_space(self):
        high = self.data_dict[next(iter(self.data_dict))].state.iloc[0]
        return spaces.Box(-high, high, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def step(self, action: float):

        done = self.is_done()

        if action == self.current_level['answer_index'].iloc[0]:
            reward = self.current_level['reward'].iloc[0]
            self.current_reward = reward
        elif action == 4:
            done = True
            reward = self.current_reward
        else:
            reward = 0
            done = True

        level = self.levels[self.iterator]
        self.current_level = self.data_dict[level].sample(1)

        state = self._get_state()
        if not done:
            return state, reward, done, {}

        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                print(
                    "You are calling 'step()' even though this environment "
                    "has already returned done = True. You should always call "
                    "'reset()' once you receive 'done = True' -- any further "
                    "steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.

        return state, reward, done, {}

    def is_done(self):
        return self._iter == self.iter_len - 1

    @property
    def iterator(self):
        self._iter = (self._iter + 1) % self.iter_len
        return self._iter

    def _get_state(self):
        return (self.current_level['state']).iloc[0]

    def reset(self):
        """
        sample emitent id
        """

        self.steps_beyond_done = None
        self.current_reward = 0
        self._iter = 0
        level = self.levels[self._iter]
        self.current_level = self.data_dict[level].sample(1)

        return self._get_state()
