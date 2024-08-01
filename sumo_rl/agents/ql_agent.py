import numpy as np
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
import torch
from torch_geometric.nn import GCNConv

class GPlightPredictor(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GPlightPredictor, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, train=self.training, p=0.5)
        x = self.conv2(x, edge_index)
        return x

class GPlight:
    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy(), gnn_model=None):
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {starting_state: [0 for _ in range(action_space.n)]}
        self.exploration = exploration_strategy
        self.acc_reward = 0
        self.gnn_model = gnn_model or GPlightPredictor(num_node_features=state_space.shape[0], num_classes=action_space.n)

    def act(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0 for _ in range(self.action_space.n)]
        if np.random.rand() < self.exploration.epsilon: # Select a random phase with probability epsilon
            self.action = np.random.choice(self.action_space.n)
        else: # Otherwise select the phase with the maximum Q-value
            self.action = np.argmax(self.q_table[state])
        return self.action

    def learn(self, state, next_state, reward, done=False):
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]
        s = state
        s1 = next_state
        a = self.action
        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
            reward + self.gamma * max(self.q_table[s1]) - self.q_table[s][a]
        )
        self.state = s1
        self.acc_reward += reward

    def predict_traffic(self, current_traffic_data):
        self.gnn_model.eval()
        with torch.no_grad():
            prediction = self.gnn_model(current_traffic_data)
        return prediction

    def adjust_green_light_duration(self, predicted_traffic, current_traffic):
        return (predicted_traffic + current_traffic) / 2

    def run_episode(self, env, T):
        t = 0
        tsum = 0
        while tsum < T:
            # Traffic prediction
            current_traffic_data = self.get_current_traffic_data()
            predicted_traffic = self.predict_traffic(current_traffic_data)

            # Traffic light control
            actions = {}
            for ts in env.ts_ids:
                actions[ts] = self.act(ts) # Use the act method to choose an action

            s, r, done, _ = env.step(actions)
            for ts in env.ts_ids:
                next_state = env.encode(s[ts], ts) # Encode the next state
                self.learn(ts, next_state, reward=r[ts])
            tsum += 1 # Increment time step
            t += 1

            if done["__all__"]:
                break

        return tsum

    def get_current_traffic_data(self):
        from torch_geometric.data import Data
        x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float) # Adjust the feature dimensions here
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        return Data(x=x, edge_index=edge_index)