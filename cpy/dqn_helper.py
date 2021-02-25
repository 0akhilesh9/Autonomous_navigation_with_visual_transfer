import abc
from collections import defaultdict
import torch
from gym import spaces
from gym.spaces.dict_space import Dict as SpaceDict
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
from habitat.config import Config
from habitat.tasks.nav.nav import (
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.models.simple_cnn import SimpleCNN
# from habitat_baselines.utils.common import CategoricalNet

class CategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return x

class Policy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions
        self.epsilon = 0.4

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )


    def forward(self, *x):
        raise NotImplementedError

    def act(self, observations, rnn_hidden_states, masks, deterministic=False):
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states, masks)
        action_scores = self.action_distribution(features)
        action = torch.reshape(action_scores.argmax(dim=1), [-1,1])

        if np.random.rand(1) < self.epsilon:
            action = torch.reshape(torch.tensor(np.random.randint(0, 3), device=features.device), [-1,1])

        # distribution = self.action_distribution(features)
        #
        # if deterministic:
        #     action = distribution.mode()
        # else:
        #     action = distribution.sample()
        #
        # action_log_probs = distribution.log_probs(action)
        #
        # return action, action_log_probs, rnn_hidden_states

        return action, action_scores, rnn_hidden_states

    def evaluate_actions(self, observations, rnn_hidden_states, prev_actions, masks, action):
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states, masks)
        distribution = self.action_distribution(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return action_log_probs, distribution_entropy, rnn_hidden_states

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space):
        pass

# @baseline_registry.register_policy
class PointNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space: SpaceDict,
        action_space_n,
        hidden_size: int = 512,
        **kwargs
    ):
        super().__init__(
            PointNavBaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                **kwargs,
            ),
            action_space_n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: SpaceDict, action_space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class PointNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: SpaceDict,
        hidden_size: int,
    ):
        super().__init__()

        if (
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            in observation_space.spaces
        ):
            self._n_input_goal = observation_space.spaces[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ].shape[0]
        elif PointGoalSensor.cls_uuid in observation_space.spaces:
            self._n_input_goal = observation_space.spaces[
                PointGoalSensor.cls_uuid
            ].shape[0]
        elif ImageGoalSensor.cls_uuid in observation_space.spaces:
            goal_observation_space = spaces.Dict(
                {"rgb": observation_space.spaces[ImageGoalSensor.cls_uuid]}
            )
            self.goal_visual_encoder = SimpleCNN(
                goal_observation_space, hidden_size
            )
            self._n_input_goal = hidden_size

        self._hidden_size = hidden_size

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size) + self._n_input_goal,
            self._hidden_size,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, masks):
        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            target_encoding = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]

        elif PointGoalSensor.cls_uuid in observations:
            target_encoding = observations[PointGoalSensor.cls_uuid]
        elif ImageGoalSensor.cls_uuid in observations:
            image_goal = observations[ImageGoalSensor.cls_uuid]
            target_encoding = self.goal_visual_encoder({"rgb": image_goal})

        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed] + x

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states


import torch
from torch import nn as nn
from torch import optim as optim

EPS_PPO = 1e-5


class DQN(nn.Module):
    def __init__(
        self, main_dqn, target_dqn, clip_param, dqn_epoch, num_mini_batch, value_loss_coef, entropy_coef, gamma=0.99, lr=None, eps=None,
        max_grad_norm=None, use_clipped_value_loss=True, use_normalized_advantage=True):

        super().__init__()

        self.main_dqn = main_dqn
        self.target_dqn = target_dqn
        self.clip_param = clip_param
        self.dqn_epoch = dqn_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, main_dqn.parameters())), lr=lr, eps=eps)
        self.device = next(main_dqn.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage
        self.gamma = gamma

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollouts):
        print("model learn")
        for _e in range(self.dqn_epoch):
            data_generator = rollouts.recurrent_generator_dqn(self.num_mini_batch)

            for sample in data_generator:
                (state_batch, actions_batch, next_state_batch, reward_batch, masks_batch,
                 recurrent_hidden_states_batch, next_masks_batch, next_recurrent_hidden_states_batch) = sample
                sample = None
                non_final_next_states = defaultdict(list)

                for sensor in next_state_batch:
                    non_final_next_states[sensor] = torch.cat([next_state_batch[sensor][i].unsqueeze(dim=0) for i in range(masks_batch.shape[0]) if masks_batch[i][0] ==1], dim=0)
                non_final_mask = [i for i in range(masks_batch.shape[0]) if masks_batch[i][0] ==1]


                action, action_scores, rnn_states = self.main_dqn.act(state_batch, recurrent_hidden_states_batch, masks_batch)
                state_values = action_scores.gather(1, actions_batch)
                next_state_values = torch.zeros(masks_batch.shape[0], device=self.device)

                action_, next_state_action_scores, rnn_states_ \
                    = self.target_dqn.act(non_final_next_states, next_recurrent_hidden_states_batch, next_masks_batch[non_final_mask])



                # next_state_values[non_final_mask] = next_state_action_scores.detach().squeeze(1)

                next_state_values.index_put([torch.tensor(non_final_mask, dtype=torch.long, device=self.device)],
                                            next_state_action_scores.detach().max(dim=1).values)

                expected_state_action_values = (next_state_values * self.gamma) + reward_batch.squeeze(1)
                total_loss = F.smooth_l1_loss(state_values, expected_state_action_values.unsqueeze(1))

                self.optimizer.zero_grad()
                total_loss.backward()
                self.before_step()
                self.optimizer.step()
                total_loss_value = total_loss.item()

                del action_
                del rnn_states_
                del actions_batch
                del non_final_next_states
                del next_state_values
                del expected_state_action_values
                del state_batch
                del masks_batch
                del next_state_batch
                del recurrent_hidden_states_batch
                del action
                del rnn_states

                torch.cuda.empty_cache()

        num_updates = self.dqn_epoch * self.num_mini_batch

        total_loss_value /= num_updates

        return total_loss_value

    def before_step(self):
        nn.utils.clip_grad_norm_(self.main_dqn.parameters(), self.max_grad_norm)