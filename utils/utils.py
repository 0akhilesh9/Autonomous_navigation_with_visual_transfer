import os
import tqdm
import imageio
import numpy as np
from typing import Dict, List, Optional




def images_to_video(images: List[np.ndarray], output_dir: str, video_name: str, fps: int = 10, quality: Optional[float] = 5, **kwargs,):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(os.path.join(output_dir, video_name),fps=fps,quality=quality,**kwargs,)
    for im in tqdm.tqdm(images):
        writer.append_data(im)
    writer.close()

def generate_video(
    video_dir: Optional[str],images: List[np.ndarray], video_name: str, fps: int = 10):
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    Returns:
        None
    """
    if len(images) < 1:
        return

    # video_name = f"episode{episode_id}_ckpt{checkpoint_idx}_{metric_name}{metric_value:.2f}"

    assert video_dir is not None
    images_to_video(images, video_dir, video_name)


def get_object_to_goal_dist_reward(env_obj, prev_dist, config):
    dist_reward = 0.0
    curr_dist = env_obj.get_metrics()["distance_to_goal"]
    if abs(curr_dist-prev_dist) < 0.2:
        dist_reward += config.TASK_REWARD.COLLISION_PENALTY
    # dist_reward = prev_dist - curr_dist + config.TASK_REWARD.LAMBDA
    dist_reward = (-1.0*curr_dist) + config.TASK_REWARD.LAMBDA
    if curr_dist <= config.TASK.SUCCESS_DISTANCE:
        dist_reward += config.TASK_REWARD.TOP_REWARD

    return dist_reward