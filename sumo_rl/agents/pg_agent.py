import os
import torch
from torch import nn
from torch.utils.data import DataLoader

class PolicyGradientAgent():
    def __init__(self, starting_state, state_space, action_space, alpha=0.5, ) -> None:
        """Init the PG Agent"""
        # what is needed, what is not?
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.policy_nn = NeuralNet()
        # don't know if we need gamma
        # self.gamma = gamma
        # no q table needed for policy gradient
        # self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        # i
        # self.exploration = exploration_strategy
        self.acc_reward = 0

    def act(self, x):
        pass

    def learn(self):
        pass


# neural net class for representing the policy.
# NOTE: we are using ReLU( Rectified Linear Unit) for non-linearity
class NeuralNet(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=8, out_features=2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, out_features),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits