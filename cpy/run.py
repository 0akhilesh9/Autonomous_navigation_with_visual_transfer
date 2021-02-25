import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import yaml
import habitat
import habitat_sim
import yacs.config
from habitat_baselines.config.default import get_config
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class

import ppo_trainer
import dqn_trainer


class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)

def get_env(task_config_file):
    task_config = habitat.get_config(task_config_file)
    env_obj = habitat.Env(task_config)

    # increase turn amount for left and right
    env_obj.sim.get_agent(0).agent_config.action_space[1].actuation.amount = 0.25
    env_obj.sim.get_agent(0).agent_config.action_space[2].actuation.amount = 10
    env_obj.sim.get_agent(0).agent_config.action_space[3].actuation.amount = 10

    return env_obj

def get_app_config(app_config_file, task_config_file):
    app_config = Config()
    config_file = r"/home/userone/workspace/bed1/project/code/config_files/ppo.yaml"
    app_config.merge_from_file(config_file)
    app_config.merge_from_file(app_config_file)
    app_config.merge_from_file(task_config_file)

    sensor_list = []
    if 'rgb_sensor' in [x.lower() for x in app_config.SIMULATOR.AGENT_0.SENSORS]:
        sensor_list.append("rgb")
    if 'depth_sensor' in [x.lower() for x in app_config.SIMULATOR.AGENT_0.SENSORS]:
        sensor_list.append("depth")
    if 'semantic_sensor' in [x.lower() for x in app_config.SIMULATOR.AGENT_0.SENSORS]:
        sensor_list.append("semantic")

    app_config.sensors_list = sensor_list

    return app_config

def train_ppo():
    task_config_file = r"/home/userone/workspace/bed1/project/code/config_files/pointnav_config_final.yaml"
    # env_obj = get_env(task_config_file)

    # app_config_file = r"/home/userone/workspace/bed1/project/code/config_files/app_config.yaml"
    # app_config = get_app_config(app_config_file, task_config_file)

    config_file = r"/home/userone/workspace/bed1/project/code/config_files/ppo_pointnav_example.yaml"
    app_config = get_config(config_file, None)

    ppo_trainer_obj = ppo_trainer.PPOTrainer(app_config)
    ppo_trainer_obj.train()

def eval_ppo():
    config_file = r"/home/userone/workspace/bed1/project/code/config_files/ppo_pointnav_example.yaml"
    app_config = get_config(config_file, None)

    ppo_trainer_obj = ppo_trainer.PPOTrainer(app_config)
    ppo_trainer_obj.eval(r"/home/userone/workspace/bed1/project/tmp/checkpoints/ppo_ckpt_0.pth")

def train_dqn():
    config_file = r"/home/userone/workspace/bed1/project/code/config_files/ppo_pointnav_example.yaml"
    app_config = get_config(config_file, None)

    dqn_trainer_obj = dqn_trainer.DQNTrainer(app_config)
    dqn_trainer_obj.train()

def eval_dqn():
    config_file = r"/home/userone/workspace/bed1/project/code/config_files/ppo_pointnav_example.yaml"
    app_config = get_config(config_file, None)

    dqn_trainer_obj = dqn_trainer.DQNTrainer(app_config)
    dqn_trainer_obj.eval(r"/home/userone/workspace/bed1/project/tmp/checkpoints/dqn_ckpt_40.pth")


if __name__ == "__main__":
    train_ppo()
    eval_ppo()
    # train_dqn()
    # eval_dqn()