from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
import numpy as np

class QLAgent:
    """pTLC Agent class."""

    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=80000, exploration_strategy=EpsilonGreedy()):
        """Initialize pTLC agent."""
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        self.exploration = EpsilonGreedy(epsilon_start, epsilon_end, epsilon_decay)
        self.acc_reward = 0
        self.pduration = 0

    def act(self):
        """Choose action based on Q-table."""
        if np.random.rand() < self.epsilon:
            self.action = np.random.choice(self.action_space.n)
        else:
            self.action = np.argmax(self.q_table[self.state])
        return self.action

    def learn(self, next_state, reward, done=False):
        """Update Q-table with new experience."""
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        s1 = next_state
        a = self.action
        best_next_action = np.argmax(self.q_table[s1])
        td_target = reward + self.gamma * self.q_table[s1][best_next_action]
        self.q_table[s][a] += self.alpha * (td_target - self.q_table[s][a])
        self.state = s1
        self.acc_reward += reward
        self.epsilon = max(self.epsilon_end, self.epsilon * np.exp(-1.0 / self.epsilon_decay))
