import os
import csv
import time
from collections import deque
import torch.nn.functional as F

from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import observations_to_image
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from core_rl.helper import batch_obs, linear_decay
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.base_trainer import BaseRLTrainer
import dqn_model
from utils.utils import generate_video, get_object_to_goal_dist_reward
from utils.buffer import ExpBuffer



class DQNTrainer():

    def __init__(self, config):
        self.main_dqn_model = None
        self.target_dqn_model = None
        self.agent = None
        self.env = None
        self._static_encoder = False
        self._encoder = None
        self.config = config
        self.device = (torch.device("cuda", self.config.TORCH_GPU_ID) if torch.cuda.is_available() else torch.device("cpu"))
        self.goal_sensor_uuid = "pointgoal_with_gps_compass"

    def init_dqn_model(self, dqn_cfg):
        self.main_dqn_model = dqn_model.Model(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            goal_sensor_uuid="pointgoal_with_gps_compass",
            device=self.device,
            hidden_size=dqn_cfg.hidden_size,
            start_epsilon=dqn_cfg.start_eps
        )
        self.target_dqn_model = dqn_model.Model(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            goal_sensor_uuid="pointgoal_with_gps_compass",
            device=self.device,
            hidden_size=dqn_cfg.hidden_size,
            start_epsilon=dqn_cfg.start_eps
        )
        self.target_dqn_model.eval()
        self.main_dqn_model.to(self.device)
        self.target_dqn_model.to(self.device)


    def _collect_rollout_step(self, rollouts, current_episode_reward, episode_rewards):
        # sample actions
        with torch.no_grad():
            step_observation = { k: v[rollouts.step] for k, v in rollouts.observations.items() }

            action, rnn_hidden_states = self.main_dqn_model(step_observation, rollouts.recurrent_hidden_states[rollouts.step], rollouts.masks[rollouts.step])


        outputs = self.env.step(action)
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        batch = batch_obs(observations)
        rewards = torch.tensor(rewards, dtype=torch.float, device=episode_rewards.device)
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor([[0.0] if done else [1.0] for done in dones], dtype=torch.float, device=episode_rewards.device)

        current_episode_reward += rewards
        episode_rewards += (1 - masks) * current_episode_reward
        current_episode_reward *= masks
        episode_dist = observations[0]['pointgoal_with_gps_compass'][0]
        done_flag = dones[0]

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        rollouts.insert(batch, action, rewards, masks)

        return episode_dist, done_flag

    def _update_agent(self, dqn_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            ).detach()

        rollouts.compute_returns(
            next_value, dqn_cfg.use_gae, dqn_cfg.gamma, dqn_cfg.tau
        )

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )

    def train(self, env):
        print("DQN training!!!")
        dqn_cfg = self.config.RL.DQN
        self.env = env
        self.init_dqn_model(dqn_cfg)

        if os.path.exists(dqn_cfg.save_path):
            self.target_dqn_model.load_state_dict(torch.load(dqn_cfg.save_path))
            self.target_dqn_model.eval()
            self.main_dqn_model.load_state_dict(torch.load(dqn_cfg.save_path))
            self.main_dqn_model.train()
            print("Model loaded from {}".format(dqn_cfg.save_path))

        rollouts = ExpBuffer(device=self.device, max_buffer_size=dqn_cfg.train.num_episodes)
        delta_epsilon = (dqn_cfg.start_eps - dqn_cfg.end_eps) / dqn_cfg.max_random_action_steps
        loss_fn = torch.nn.MSELoss()
        # self.optimizer = torch.optim.Adam(
        #     list(filter(lambda p: p.requires_grad, self.main_dqn_model.parameters())), lr=float(dqn_cfg.lr), eps=float(dqn_cfg.eps))

        self.optimizer = torch.optim.Adam(self.main_dqn_model.parameters())

        # (num_recurrent_layers, batch, recurrent_hidden_state_size)
        recurrent_hidden_states_init = torch.zeros(1, 1, dqn_cfg.hidden_size).to(device=self.device)
        masks_init = torch.zeros(1, 1).to(device=self.device)
        episode_reward_list = []
        episode_dist_list = []
        episode_spl_list = []
        episode_success_list = []
        loss_list = []

        for episode_count in range(dqn_cfg.train.num_episodes):
            recurrent_hidden_states = recurrent_hidden_states_init
            masks = masks_init
            print("Episode {}\n".format(episode_count))
            episode_buffer = []
            rgb_frames = []
            episode_reward = 0.0

            observations = self.env.reset()
            prev_dist = self.env.get_metrics()["distance_to_goal"]
            current_step = 0

            lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lambda x: 1 - (x / float(dqn_cfg.train.num_episodes)))

            # to convert dtype of ndarray
            for sensor in observations:
                observations[sensor] = observations[sensor].astype('float64')

            pre_batch_observation = batch_obs(observations, device=self.device)

            while current_step < dqn_cfg.train.max_steps:

                # if dqn_cfg.use_linear_clip_decay:
                #     self.agent.clip_param = dqn_cfg.clip_param * linear_decay(current_step, dqn_cfg.train.max_steps)

                with torch.no_grad():
                    action, rnn_hidden_states, action_probs = self.main_dqn_model(pre_batch_observation, [recurrent_hidden_states, masks], 1)

                observations = self.env.step(action.tolist())

                dones = [env.get_metrics()["success"]]
                # to convert dtype of ndarray
                for sensor in observations:
                    observations[sensor] = observations[sensor].astype('float64')
                # observations[self.goal_sensor_uuid] =
                rewards = get_object_to_goal_dist_reward(self.env, prev_dist, self.config)
                delta_dist = abs(self.env.get_metrics()["distance_to_goal"] - prev_dist)
                prev_dist = self.env.get_metrics()["distance_to_goal"]
                episode_reward += rewards

                post_batch_observation = batch_obs(observations, device=self.device)
                rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
                masks = torch.tensor([[0.0] if done else [1.0] for done in dones], dtype=torch.float, device=self.device)

                episode_buffer.append([pre_batch_observation, action, rewards, post_batch_observation, masks])

                sample_data = rollouts.sample(dqn_cfg.batch_size, dqn_cfg.rnn_seq_len)
                if len(sample_data) >0:
                    action, rnn_hidden_states_, action_probs = self.main_dqn_model(sample_data[3], [], dqn_cfg.rnn_seq_len)

                    # with torch.no_grad():
                    action_, rnn_hidden_states_, q_tmp_target = self.target_dqn_model(sample_data[3], [], dqn_cfg.rnn_seq_len)
                    action_ = action_.detach()
                    rnn_hidden_states_ = rnn_hidden_states_.detach()
                    q_tmp_target = q_tmp_target.detach()

                    q_tmp_target = torch.gather(q_tmp_target, -1, (action-1).unsqueeze(1)).reshape(-1)
                    q_target = sample_data[2].reshape(-1) + (dqn_cfg.discount_factor * q_tmp_target * sample_data[4])
                    action_one_hot = torch.nn.functional.one_hot(sample_data[1]-1, num_classes=self.env.action_space.n-1)
                    q_predicted = torch.sum(action_one_hot * action_probs, dim=1)
                    # loss = torch.mean(loss_fn(q_predicted, q_target))
                    loss = F.smooth_l1_loss(q_predicted, q_target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.main_dqn_model.parameters(), dqn_cfg.max_grad_norm)
                    for param in self.main_dqn_model.parameters():
                        param.grad.data.clamp_(-1, 1)
                    self.optimizer.step()
                    print("loss: {}".format(loss.tolist()))
                    print(q_predicted)
                    print(q_target)
                    loss_list.append(loss.tolist())
                    if dqn_cfg.use_linear_lr_decay:
                        lr_scheduler.step()
                else:
                    print("Low sample size. Skipped model update!!!")

                current_step += 1
                pre_batch_observation = post_batch_observation
                recurrent_hidden_states = rnn_hidden_states

                if dones[0]:
                    print("Episode Over!!!")
                    break
                else:
                    # frame = observations_to_image(observations, {})
                    # top_down_map = maps.get_topdown_map_from_sim(self.env.sim, map_resolution=frame.shape[0])
                    # recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
                    # top_down_map = recolor_map[top_down_map]
                    # frame = np.concatenate((frame, top_down_map.transpose(1,0,2)), axis=1)
                    top_down_map = env.get_metrics()['top_down_map']
                    frame = observations_to_image(observations, {"top_down_map": top_down_map})
                    rgb_frames.append(frame.astype(np.uint8))

                if current_step <= dqn_cfg.max_random_action_steps:
                    self.main_dqn_model.update_epsilon(delta_epsilon, dqn_cfg.end_eps)

                if (current_step % dqn_cfg.train.update_frequency) == 0:
                    self.target_dqn_model.load_state_dict(self.main_dqn_model.state_dict())
                    self.target_dqn_model.eval()

            if len(dqn_cfg.train.VIDEO_OPTION) > 0:
                generate_video(video_dir=self.config.VIDEO_DIR, images=rgb_frames, video_name="train_episode_{}".format(episode_count))

            print(len(episode_buffer))
            rgb_frames = []
            rollouts.add(episode_buffer)
            dist = self.env.get_metrics()["distance_to_goal"]
            episode_dist_list.append(dist)
            episode_spl_list.append(self.env.get_metrics()["spl"])
            episode_success_list.append(self.env.get_metrics()["success"])
            episode_reward_list.append(episode_reward)
            print("Episode reward: {}   Distance from goal: {}".format(episode_reward, dist))

        self.target_dqn_model.load_state_dict(self.main_dqn_model.state_dict())
        torch.save(self.target_dqn_model.state_dict(), dqn_cfg.save_path)

        with open(self.config.RL.DQN.train.out_filename, 'w') as file_obj:
            writer = csv.writer(file_obj)
            columns = ["reward", "success", "spl", "dist"]
            writer.writerow(columns)
            data_rows = [[episode_reward_list[i], episode_success_list[i], episode_spl_list[i], episode_dist_list[i]] for i in range(len(episode_reward_list))]
            writer.writerows(data_rows)

        with open(self.config.RL.DQN.train.loss_filename, 'w') as file_obj:
            writer = csv.writer(file_obj)
            columns = ["loss"]
            writer.writerow(columns)
            data_rows = [[loss_list[i]] for i in range(len(loss_list))]
            writer.writerows(data_rows)

        self.env.close()

    def eval(self, env):
        print("DQN evaluation!!!")
        dqn_cfg = self.config.RL.DQN
        self.env = env
        self.init_dqn_model(dqn_cfg)

        if os.path.exists(dqn_cfg.save_path):
            self.target_dqn_model.load_state_dict(torch.load(dqn_cfg.save_path))
            self.target_dqn_model.eval()

        # (num_recurrent_layers, batch, recurrent_hidden_state_size)
        recurrent_hidden_states_init = torch.zeros(1, 1, dqn_cfg.hidden_size).to(device=self.device)
        masks_init = torch.zeros(1, 1).to(device=self.device)
        episode_reward_list = []
        episode_dist_list = []
        episode_spl_list = []
        episode_success_list = []

        for episode_count in range(dqn_cfg.test.num_episodes):
            recurrent_hidden_states = recurrent_hidden_states_init
            masks = masks_init
            print("Episode {}\n".format(episode_count))
            rgb_frames = []
            episode_reward = 0.0

            observations = self.env.reset()
            current_step = 0

            # to convert dtype of ndarray
            for sensor in observations:
                observations[sensor] = observations[sensor].astype('float64')

            pre_batch_observation = batch_obs(observations, device=self.device)

            while current_step < dqn_cfg.test.max_steps:

                action, rnn_hidden_states, action_probs = self.target_dqn_model(pre_batch_observation,
                                                                                  [recurrent_hidden_states, masks], 1)

                observations = self.env.step(action.tolist())

                dones = [env.get_metrics()["success"]]
                # to convert dtype of ndarray
                for sensor in observations:
                    observations[sensor] = observations[sensor].astype('float64')
                rewards = get_object_to_goal_dist_reward(self.env, dist, self.config)
                episode_reward += rewards


                post_batch_observation = batch_obs(observations, device=self.device)
                rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
                masks = torch.tensor([[0.0] if done else [1.0] for done in dones], dtype=torch.float,
                                     device=self.device)

                current_step += 1
                pre_batch_observation = post_batch_observation
                recurrent_hidden_states = rnn_hidden_states

                if dones[0]:
                    print("Episode Over!!!")
                    break
                else:
                    top_down_map = env.get_metrics()['top_down_map']
                    frame = observations_to_image(observations, {"top_down_map": top_down_map})
                    rgb_frames.append(frame.astype(np.uint8))

            if len(dqn_cfg.test.VIDEO_OPTION) > 0:
                generate_video(video_dir=self.config.VIDEO_DIR, images=rgb_frames, video_name="eval_episode_{}".format(episode_count))
            rgb_frames = []
            dist = self.env.get_metrics()["distance_to_goal"]
            episode_dist_list.append(dist)
            episode_spl_list.append(self.env.get_metrics()["spl"])
            episode_success_list.append(self.env.get_metrics()["success"])
            episode_reward_list.append(episode_reward)
            print("Episode reward: {}   Distance from goal: {}".format(episode_reward, dist))

        with open(self.config.RL.DQN.test.out_filename, 'w') as file_obj:
            writer = csv.writer(file_obj)
            columns = ["reward", "success", "spl", "dist"]
            writer.writerow(columns)
            data_rows = [[episode_reward_list[i], episode_success_list[i], episode_spl_list[i], episode_dist_list[i]] for i in range(len(episode_reward_list))]
            writer.writerows(data_rows)

        self.env.close()



    # def eval(self, checkpoint_path) :
    #     r"""Evaluates a single checkpoint.
    #
    #     Args:
    #         checkpoint_path: path of checkpoint
    #         writer: tensorboard writer object for logging to tensorboard
    #         checkpoint_index: index of cur checkpoint for logging
    #
    #     Returns:
    #         None
    #     """
    #     self.device = (
    #         torch.device("cuda", self.config.TORCH_GPU_ID)
    #         if torch.cuda.is_available()
    #         else torch.device("cpu")
    #     )
    #     # Map location CPU is almost always better than mapping to a CUDA device.
    #     ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
    #
    #     if self.config.EVAL.USE_CKPT_CONFIG:
    #         config = self._setup_eval_config(ckpt_dict["config"])
    #     else:
    #         config = self.config.clone()
    #
    #     dqn_cfg = config.RL.DQN
    #
    #     config.defrost()
    #     config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
    #     config.freeze()
    #
    #     if len(self.config.VIDEO_OPTION) > 0:
    #         config.defrost()
    #         config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    #         config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
    #         config.freeze()
    #
    #     self.env = construct_envs(config, get_env_class(config.ENV_NAME))
    #     self._setup_actor_critic_agent(dqn_cfg)
    #
    #     self.agent.load_state_dict(ckpt_dict["state_dict"])
    #     self.actor_critic = self.agent.actor_critic
    #
    #     # get name of performance metric, e.g. "spl"
    #     metric_name = self.config.TASK_CONFIG.TASK.MEASUREMENTS[0]
    #     metric_cfg = getattr(self.config.TASK_CONFIG.TASK, metric_name)
    #     measure_type = baseline_registry.get_measure(metric_cfg.TYPE)
    #     assert measure_type is not None, "invalid measurement type {}".format(
    #         metric_cfg.TYPE
    #     )
    #     self.metric_uuid = measure_type(sim=None, task=None, config=None)._get_uuid()
    #
    #     observations = self.env.reset()
    #     batch = batch_obs(observations, self.device)
    #
    #     current_episode_reward = torch.zeros(
    #         self.env.num_envs, 1, device=self.device
    #     )
    #
    #     test_recurrent_hidden_states = torch.zeros(
    #         self.actor_critic.net.num_recurrent_layers,
    #         self.config.NUM_PROCESSES,
    #         dqn_cfg.hidden_size,
    #         device=self.device,
    #     )
    #     prev_actions = torch.zeros(
    #         self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
    #     )
    #     not_done_masks = torch.zeros(
    #         self.config.NUM_PROCESSES, 1, device=self.device
    #     )
    #     stats_episodes = dict()  # dict of dicts that stores stats per episode
    #
    #     rgb_frames = [
    #         [] for _ in range(self.config.NUM_PROCESSES)
    #     ]  # type: List[List[np.ndarray]]
    #     if len(self.config.VIDEO_OPTION) > 0:
    #         os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
    #
    #     self.actor_critic.eval()
    #     while (
    #         len(stats_episodes) < self.config.TEST_EPISODE_COUNT
    #         and self.env.num_envs > 0
    #     ):
    #         current_episodes = self.env.current_episodes()
    #
    #         with torch.no_grad():
    #             (
    #                 _,
    #                 actions,
    #                 _,
    #                 test_recurrent_hidden_states,
    #             ) = self.actor_critic.act(
    #                 batch,
    #                 test_recurrent_hidden_states,
    #                 prev_actions,
    #                 not_done_masks,
    #                 deterministic=False,
    #             )
    #
    #             prev_actions.copy_(actions)
    #
    #         outputs = self.env.step([a[0].item() for a in actions])
    #
    #         observations, rewards, dones, infos = [
    #             list(x) for x in zip(*outputs)
    #         ]
    #         batch = batch_obs(observations, self.device)
    #
    #         not_done_masks = torch.tensor(
    #             [[0.0] if done else [1.0] for done in dones],
    #             dtype=torch.float,
    #             device=self.device,
    #         )
    #
    #         rewards = torch.tensor(
    #             rewards, dtype=torch.float, device=self.device
    #         ).unsqueeze(1)
    #         current_episode_reward += rewards
    #         next_episodes = self.env.current_episodes()
    #         envs_to_pause = []
    #         n_envs = self.env.num_envs
    #         for i in range(n_envs):
    #             if (
    #                 next_episodes[i].scene_id,
    #                 next_episodes[i].episode_id,
    #             ) in stats_episodes:
    #                 envs_to_pause.append(i)
    #
    #             # episode ended
    #             if not_done_masks[i].item() == 0:
    #                 episode_stats = dict()
    #                 episode_stats[self.metric_uuid] = infos[i][
    #                     self.metric_uuid
    #                 ]
    #                 episode_stats["success"] = int(
    #                     infos[i][self.metric_uuid] > 0
    #                 )
    #                 episode_stats["reward"] = current_episode_reward[i].item()
    #                 current_episode_reward[i] = 0
    #                 # use scene_id + episode_id as unique id for storing stats
    #                 stats_episodes[
    #                     (
    #                         current_episodes[i].scene_id,
    #                         current_episodes[i].episode_id,
    #                     )
    #                 ] = episode_stats
    #
    #                 if len(self.config.VIDEO_OPTION) > 0:
    #                     generate_video(
    #                         video_option=self.config.VIDEO_OPTION,
    #                         video_dir=self.config.VIDEO_DIR,
    #                         images=rgb_frames[i],
    #                         episode_id=current_episodes[i].episode_id,
    #                         checkpoint_idx=0,
    #                         metric_name=self.metric_uuid,
    #                         metric_value=infos[i][self.metric_uuid],
    #                     )
    #
    #                     rgb_frames[i] = []
    #
    #             # episode continues
    #             elif len(self.config.VIDEO_OPTION) > 0:
    #                 frame = observations_to_image(observations[i], infos[i])
    #                 rgb_frames[i].append(frame)
    #
    #         (
    #             self.env,
    #             test_recurrent_hidden_states,
    #             not_done_masks,
    #             current_episode_reward,
    #             prev_actions,
    #             batch,
    #             rgb_frames,
    #         ) = self._pause_envs(
    #             envs_to_pause,
    #             self.env,
    #             test_recurrent_hidden_states,
    #             not_done_masks,
    #             current_episode_reward,
    #             prev_actions,
    #             batch,
    #             rgb_frames,
    #         )
    #
    #     aggregated_stats = dict()
    #     for stat_key in next(iter(stats_episodes.values())).keys():
    #         aggregated_stats[stat_key] = sum(
    #             [v[stat_key] for v in stats_episodes.values()]
    #         )
    #     num_episodes = len(stats_episodes)
    #
    #     episode_reward_mean = aggregated_stats["reward"] / num_episodes
    #     episode_metric_mean = aggregated_stats[self.metric_uuid] / num_episodes
    #     episode_success_mean = aggregated_stats["success"] / num_episodes
    #
    #     print(f"Average episode reward: {episode_reward_mean:.6f}")
    #     print(f"Average episode success: {episode_success_mean:.6f}")
    #     print(
    #         f"Average episode {self.metric_uuid}: {episode_metric_mean:.6f}"
    #     )
    #
    #
    #     if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
    #         step_id = ckpt_dict["extra_state"]["step"]
    #
    #     print(
    #         "eval_reward", {"average reward": episode_reward_mean}
    #     )
    #     print(
    #         f"eval_{self.metric_uuid}",
    #         {f"average {self.metric_uuid}": episode_metric_mean},
    #     )
    #     print(
    #         "eval_success", {"average success": episode_success_mean}
    #     )
    #
    #     self.env.close()

