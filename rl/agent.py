import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SimpleNNAgent(nn.Module):
    def __init__(self, obs_shape, n_actions, reuse=False):
        """A simple actor-critic agent"""
        super(self.__class__, self).__init__()

        self.conv1 = nn.Conv1d(obs_shape[0], 16, kernel_size=5)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=5)
        self.maxpool_1 = nn.MaxPool1d(2)
        self.batch_norm_1 = nn.BatchNorm1d(num_features=16)
        self.batch_norm_2 = nn.BatchNorm1d(num_features=16)
        self.batch_norm_3 = nn.BatchNorm1d(num_features=16)

        self.conv3 = nn.Conv1d(16, 16, kernel_size=5)
        self.maxpool_2 = nn.MaxPool1d(2)
        self.maxpool_3 = nn.MaxPool1d(2)

        self.flatten = Flatten()
        self.hid = nn.Linear(1472, 128)
        self.logits = nn.Linear(128, n_actions)
        self.state_value = nn.Linear(128, 1)

    def forward(self, obs_t):
        """
        Takes agent's previous step and observation,
        returns next state and whatever it needs to learn (tf tensors)
        """
        x = F.elu(self.conv1(obs_t))
        x = self.maxpool_1(x)
        x = self.batch_norm_1(x)
        x = F.elu(self.conv2(x))
        x = self.maxpool_2(x)
        x = self.batch_norm_2(x)

        x = F.elu(self.conv3(x))
        x = self.maxpool_3(x)
        x = self.batch_norm_3(x)
        x = self.flatten(x)
        embedded = F.relu(self.hid(x))
        # embedded = self.dropout(embedded)

        logits = self.logits(embedded)
        state_value = self.state_value(embedded)
        return (logits, state_value)

    def sample_actions(self, agent_outputs):
        """pick actions given numeric agent outputs (np arrays)"""
        logits, state_values = agent_outputs
        probs = F.softmax(logits, dim=1)
        return torch.multinomial(probs, 1)[:, 0].data.numpy()

    def step(self, obs_t):
        """ like forward, but obs_t is a numpy array """
        obs_t = torch.tensor(np.asarray(obs_t), dtype=torch.float32)
        (l, s) = self.forward(obs_t)
        return (l.detach(), s.detach())

    def sample_greedy(self, obs_t):
        """pick actions given numeric agent outputs (np arrays)"""
        agent_outputs = self.step(obs_t[None, ...])
        logits, state_values = agent_outputs
        probs = float(torch.argmax(logits, dim=1)[0])
        return probs