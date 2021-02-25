import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import yaml
import habitat
import habitat_sim
import yacs.config
from habitat_baselines.config.default import get_config
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class

from dqn_trainer import DQNTrainer


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

def train_dqn():
    task_config_file = r"/home/userone/workspace/bed1/project/code/config_files/pointnav_config_final.yaml"
    env_obj = get_env(task_config_file)

    app_config_file = r"/home/userone/workspace/bed1/project/code/config_files/app_config.yaml"
    app_config = get_app_config(app_config_file, task_config_file)
    dqn_trainer = DQNTrainer(app_config)
    dqn_trainer.train(env_obj)

def eval_dqn():
    task_config_file = r"/home/userone/workspace/bed1/project/code/config_files/pointnav_eval_task.yaml"
    env_obj = get_env(task_config_file)

    app_config_file = r"/home/userone/workspace/bed1/project/code/config_files/app_config.yaml"
    app_config = get_app_config(app_config_file, task_config_file)
    dqn_trainer = DQNTrainer(app_config)
    dqn_trainer.eval(env_obj)




if __name__ == '__main__':

    # config = get_config("/home/userone/workspace/bed1/project/code/config_files/abc.yaml")
    # env = construct_envs(config, get_env_class(config.ENV_NAME))
    # env.reset()

    #### 1
    # config = habitat.get_config(config_paths="/home/userone/workspace/bed1/project/habitat-lab/configs/tasks/pointnav.yaml")
    # config = habitat.get_config(config_paths=config_file)
    # env = habitat.Env(config=config)

    #### 2
    # config_file = r"/home/userone/workspace/bed1/project/code/config_files/ppo_pointnav_example.yaml"
    # config = habitat.get_config(config_file, None)
    # env = construct_envs(config, get_env_class(config.ENV_NAME))
    # env.reset()

    # config.defrost()
    # config.TASK_CONFIG.DATASET.DATA_PATH = config.DATA_PATH_SET
    # config.TASK_CONFIG.DATASET.SCENES_DIR = config.SCENE_DIR_SET
    # config.freeze()

    train_dqn()
    # eval_dqn()

