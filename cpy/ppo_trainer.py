import os
import csv
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)

from rollout_str import RolloutStorage
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.utils.common import (
    batch_obs,
    linear_decay,
)
from habitat_baselines.utils.env_utils import construct_envs

import sys
sys.path.append(".")
from utils.utils import generate_video
import policy as cust_policy


@baseline_registry.register_trainer(name="ppo")
class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        self.obs_transforms = []
        self.loss_list = []
        self.episode_reward_list = []
        self.episode_dist_list = []
        self.episode_spl_list = []
        self.episode_step_list = []
        self._encoder = None

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        self.obs_space = self.envs.observation_spaces[0]
        self.actor_critic = cust_policy.PointNavBaselinePolicy(
            observation_space=self.envs.observation_spaces[0],
            action_space_n=self.envs.action_spaces[0].n-1,
            hidden_size=ppo_cfg.hidden_size,
            # goal_sensor_uuid="pointgoal_with_gps_compass",
        )

        self.actor_critic.to(self.device)

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    def save_checkpoint(self, file_name, extra_state: Optional[Dict] = None):
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name))

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

    def _collect_rollout_step(self, rollouts, current_episode_reward):
        # sample actions
        with torch.no_grad():
            step_observation = {k: v[rollouts.step] for k, v in rollouts.observations.items()}

            (values, actions, actions_log_probs, recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_observation, rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.masks[rollouts.step])

        outputs = self.envs.step([a[0].item()+1 for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        self.episode_step_count += 1
        self.episode_done = dones[0]

        # to convert dtype of ndarray
        for i in range(len(observations)):
            for sensor_data in observations[i]:
                observations[i][sensor_data] = observations[i][sensor_data].astype('float64')

        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        rewards = torch.tensor(rewards, dtype=torch.float, device=current_episode_reward.device)
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor([[0.0] if done else [1.0] for done in dones], dtype=torch.float, device=current_episode_reward.device)

        if len(self.config.RL.PPO.train.VIDEO_OPTION) > 0:
            top_down_map = infos[0]['top_down_map']
            frame = observations_to_image(observations[0], {"top_down_map": top_down_map})
            self.rgb_frames.append(frame.astype(np.uint8))

        self.episode_cumulative_reward += rewards.tolist()[0][0]
        self.dist_to_goal = infos[0]["distance_to_goal"]
        self.spl = infos[0]["spl"]
        current_episode_reward += rewards
        current_episode_reward *= masks
        rollouts.insert(batch,recurrent_hidden_states,actions,actions_log_probs,values,rewards,masks)

    def _update_agent(self, ppo_cfg, rollouts):
        with torch.no_grad():
            last_observation = {k: v[rollouts.step] for k, v in rollouts.observations.items()}
            next_value = self.actor_critic.get_value(
                last_observation, rollouts.recurrent_hidden_states[rollouts.step], rollouts.prev_actions[rollouts.step], rollouts.masks[rollouts.step]).detach()

        rollouts.compute_returns(next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau)
        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)
        rollouts.after_update()

        return value_loss, action_loss

    def train(self):
        print("Actor-Critic model training")
        self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))

        ppo_cfg = self.config.RL.PPO
        self.device = (torch.device("cuda", self.config.TORCH_GPU_ID) if torch.cuda.is_available() else torch.device("cpu"))
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(ppo_cfg)

        lr_scheduler = LambdaLR(optimizer=self.agent.optimizer, lr_lambda=lambda x: linear_decay(x, self.config.RL.PPO.train.num_episodes))

        rollouts = RolloutStorage(ppo_cfg.train.max_steps, self.envs.num_envs, self.obs_space, self.envs.action_spaces[0], ppo_cfg.hidden_size)
        rollouts.to(self.device)

        # episode stats
        self.episode_count = 0
        self.episode_cumulative_reward = 0.0
        self.dist_to_goal = 9999.0
        self.spl = 0.0
        self.episode_step_count = 0
        self.rgb_frames = []
        self.episode_done = False

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        count_steps = 0
        count_checkpoints = 0

        for update in range(self.config.RL.PPO.train.num_episodes):
            observations = self.envs.reset()
            # to convert dtype of ndarray
            for i in range(len(observations)):
                for sensor_data in observations[i]:
                    observations[i][sensor_data] = observations[i][sensor_data].astype('float64')

            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            for sensor in rollouts.observations:
                rollouts.observations[sensor][0].copy_(batch[sensor])

            # batch and observations may contain shared PyTorch CUDA
            # tensors.  We must explicitly clear them here otherwise
            # they will be kept in memory for the entire duration of training!
            batch = None
            observations = None
            print("Episode {} running".format(update))

            if ppo_cfg.use_linear_lr_decay:
                lr_scheduler.step()

            if ppo_cfg.use_linear_clip_decay:
                self.agent.clip_param = ppo_cfg.clip_param * linear_decay(update, self.config.RL.PPO.train.num_episodes)

            for _step in range(ppo_cfg.train.max_steps):
                self._collect_rollout_step(rollouts, current_episode_reward)
                if self.episode_done:
                    # update episode stats
                    self.episode_reward_list.append(self.episode_cumulative_reward)
                    self.episode_dist_list.append(self.dist_to_goal)
                    self.episode_spl_list.append(self.spl)
                    self.episode_step_list.append(self.episode_step_count)
                    if len(self.config.RL.PPO.train.VIDEO_OPTION) > 0:
                        generate_video(video_dir=self.config.VIDEO_DIR, images=self.rgb_frames, video_name="ppo_train_episode_{}".format(self.episode_count))

                    # reset episode stats
                    self.episode_cumulative_reward = 0.0
                    self.dist_to_goal = 9999.0
                    self.spl = 0.0
                    self.episode_step_count = 0
                    self.rgb_frames = []
                    self.episode_done = False
                    self.episode_count += 1

            value_loss, action_loss = self._update_agent(ppo_cfg, rollouts)

            losses = [value_loss, action_loss]
            total_loss_val = sum(losses)

            # checkpoint model
            if (update % self.config.RL.PPO.train.checkpoint_save) == 0:
                self.save_checkpoint(f"ppo_ckpt_{update}.pth", dict(step=count_steps))
                count_checkpoints += 1

            self.loss_list.append(total_loss_val)
            print("loss: {}".format(total_loss_val))

        self.envs.close()

        with open(self.config.RL.PPO.train.out_filename, 'w') as file_obj:
            writer = csv.writer(file_obj)
            columns = ["cumulative_reward", "step_count", "spl", "dist"]
            writer.writerow(columns)
            data_rows = [[self.episode_reward_list[i], self.episode_step_list[i], self.episode_spl_list[i], self.episode_dist_list[i]] for i
                         in range(len(self.episode_reward_list))]
            writer.writerows(data_rows)

        with open(self.config.RL.PPO.train.loss_filename, 'w') as file_obj:
            writer = csv.writer(file_obj)
            columns = ["loss"]
            writer.writerow(columns)
            data_rows = [[self.loss_list[i]] for i in range(len(self.loss_list))]
            writer.writerows(data_rows)

    def eval(self, checkpoint_path):
        print("Actor-Critic evaluation")
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        self.device = (torch.device("cuda", self.config.TORCH_GPU_ID) if torch.cuda.is_available() else torch.device("cpu"))
        config = self.config.clone()
        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
        config.freeze()

        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self._setup_actor_critic_agent(ppo_cfg)
        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic
        self.actor_critic.eval()

        episode_reward_list = []
        episode_dist_list = []
        episode_spl_list = []
        episode_step_list = []

        for episode_idx in range(self.config.RL.PPO.test.num_episodes):
            print("Episode {} running".format(episode_idx))
            observations = self.envs.reset()
            # to convert dtype of ndarray
            for i in range(len(observations)):
                for sensor_data in observations[i]:
                    observations[i][sensor_data] = observations[i][sensor_data].astype('float32')

            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            test_recurrent_hidden_states = torch.zeros(self.actor_critic.net.num_recurrent_layers,
                                                       self.config.NUM_PROCESSES, ppo_cfg.hidden_size,
                                                       device=self.device)
            not_done_masks = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device)
            rgb_frames = []
            episode_cumulative_reward = 0.0
            episode_done = False
            dist_to_goal = 9999
            spl = 0.0
            step_count = 0

            # Runs till max_steps specified in config file
            while episode_done==False:
                with torch.no_grad():
                    (_, actions, _, test_recurrent_hidden_states,
                    ) = self.actor_critic.act(batch, test_recurrent_hidden_states, not_done_masks, deterministic=False)

                outputs = self.envs.step([a[0].item()+1 for a in actions])
                step_count += 1

                observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
                # to convert dtype of ndarray
                for i in range(len(observations)):
                    for sensor_data in observations[i]:
                        observations[i][sensor_data] = observations[i][sensor_data].astype('float32')

                batch = batch_obs(observations, device=self.device)
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)
                episode_cumulative_reward += rewards[0]
                dist_to_goal = infos[0]["distance_to_goal"]
                spl = infos[0]["spl"]
                episode_done = dones[0]

                if len(self.config.RL.PPO.test.VIDEO_OPTION) > 0:
                    frame = observations_to_image({k: v[i] for k, v in batch.items()}, infos[i])
                    rgb_frames.append(frame)

            episode_reward_list.append(episode_cumulative_reward)
            episode_step_list.append(step_count)
            episode_dist_list.append(dist_to_goal)
            episode_spl_list.append(spl)
            if len(self.config.VIDEO_OPTION) > 0:
                generate_video(video_dir=self.config.VIDEO_DIR, images=rgb_frames, video_name="ppo_eval_episode_{}".format(episode_idx))

        with open(self.config.RL.PPO.test.out_filename, 'w') as file_obj:
            writer = csv.writer(file_obj)
            columns = ["cumulative_reward", "step_count", "spl", "dist"]
            writer.writerow(columns)
            data_rows = [[episode_reward_list[i], episode_step_list[i], episode_spl_list[i], episode_dist_list[i]] for i
                         in range(len(episode_reward_list))]
            writer.writerows(data_rows)

        self.envs.close()