import torch
import torch.nn as nn
import numpy as np

from core_rl import cnn, rnn, helper

class Model(nn.Module):
    def __init__(self,observation_space,action_space,goal_sensor_uuid, device, hidden_size=512, start_epsilon=0.01):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self.num_input_goal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
        self.hidden_size = hidden_size
        self.epsilon = start_epsilon
        self.device = device

        # cnn layer
        self.cnn_model = cnn.SimpleCNN(observation_space, hidden_size)
        # rnn layer
        self.rnn_encoder = rnn.RNNStateEncoder(self.hidden_size, self.hidden_size)
        # linear layer
        self.linear = nn.Linear(self.hidden_size + self.num_input_goal, action_space.n-1)  # we are not considering stop action (0 - stop)
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)
        # set model to train mode
        self.train()

    def forward(self, observations, rnn_params, seq_len=1):
        cnn_features = self.cnn_model(observations)
        # rnn_input = torch.cat([cnn_features, observations[self.goal_sensor_uuid]], dim=1)
        rnn_input = torch.reshape(cnn_features, [-1, seq_len, self.hidden_size])

        if len(rnn_params) > 0:
            rnn_output, rnn_hidden_states = self.rnn_encoder(rnn_input, rnn_params)
        else:
            rnn_output, rnn_hidden_states = self.rnn_encoder(rnn_input, [])
        new_shape = list(rnn_output.shape[:2])
        new_shape.append(-1)
        rnn_output = torch.cat([rnn_output,torch.reshape(observations[self.goal_sensor_uuid], new_shape)], dim=2)
        action_probs = self.linear(rnn_output.reshape(-1,self.hidden_size + self.num_input_goal))

        if len(rnn_params) > 0:
            if np.random.rand(1) < self.epsilon:
                action = torch.tensor(np.random.randint(1, 4), device=self.device)
            else:
                action = torch.argmax(action_probs, dim=1) + 1
                action = action.squeeze()
        else:
            action = torch.argmax(action_probs, dim=1) + 1

        return action, rnn_hidden_states, action_probs

    def update_epsilon(self, new_value, end_eps):
        if (self.epsilon-new_value) > end_eps:
            self.epsilon = self.epsilon - new_value
        else:
            self.epsilon = end_eps

    def evaluate_actions(self, observations, rnn_hidden_states, masks, action):
        features, rnn_hidden_states = self.forward_pass(observations, rnn_hidden_states, masks)
        distribution = self.action_distribution(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return action_log_probs, distribution_entropy, rnn_hidden_states


class Model1(nn.Module):
    def __init__(self,observation_space,action_space,goal_sensor_uuid, device, hidden_size=512, start_epsilon=0.01):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self.num_input_goal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
        self.hidden_size = hidden_size
        self.epsilon = start_epsilon
        self.device = device

        # cnn layer
        self.cnn_model = cnn.SimpleCNN(observation_space, hidden_size)
        # rnn layer
        self.rnn_encoder = rnn.RNNStateEncoder(self.hidden_size + self.num_input_goal, self.hidden_size)
        # linear layer
        self.linear = nn.Linear(self.hidden_size, action_space.n-1)  # we are not considering stop action (0 - stop)
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)
        # set model to train mode
        self.train()

    def forward(self, observations, rnn_params, seq_len=1):
        cnn_features = self.cnn_model(observations)
        rnn_input = torch.cat([cnn_features, observations[self.goal_sensor_uuid]], dim=1)
        rnn_input = torch.reshape(rnn_input, [-1, seq_len, self.hidden_size + self.num_input_goal])

        if len(rnn_params) > 0:
            rnn_output, rnn_hidden_states = self.rnn_encoder(rnn_input, rnn_params)
        else:
            rnn_output, rnn_hidden_states = self.rnn_encoder(rnn_input, [])

        action_probs = self.linear(rnn_output.reshape(-1,self.hidden_size))

        if len(rnn_params) > 0:
            if np.random.rand(1) < self.epsilon:
                action = torch.tensor(np.random.randint(1, 4), device=self.device)
            else:
                action = torch.argmax(action_probs, dim=1) + 1
                action = action.squeeze()
        else:
            action = torch.argmax(action_probs, dim=1) + 1

        return action, rnn_hidden_states, action_probs

    def update_epsilon(self, new_value):
        if (self.epsilon-new_value) > 0:
            self.epsilon = self.epsilon - new_value

    def evaluate_actions(self, observations, rnn_hidden_states, masks, action):
        features, rnn_hidden_states = self.forward_pass(observations, rnn_hidden_states, masks)
        distribution = self.action_distribution(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return action_log_probs, distribution_entropy, rnn_hidden_states


