import warnings
from collections.abc import Callable

from minigrid.wrappers import ImgObsWrapper

try:
    PROCGEN_AVAILABLE = True
    import gym
    from procgen import ProcgenEnv
except ModuleNotFoundError:
    PROCGEN_AVAILABLE = False
    warnings.warn("gym package not found. Can't run ProcGen!", ImportWarning)

import gymnasium
import numpy as np

from .atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from .minigrid_wrappers import (
    ChannelFirstObsWrapper,
    DenseRewardWrapper,
    LavaNegativeRewardWrapper,
    LavaNotDeadWrapper,
    NoOrientationActionWrapper,
)
from .procgen_wrappers import ResizeRenderWrapper

if PROCGEN_AVAILABLE:

    def make_procgen(
        env_id: str,
        num_envs: int,
        level_distribution: str,
        start_level: int,
        num_levels: int,
        capture_video: bool,
        gamma: float,
        run_name: str,
    ) -> ProcgenEnv:
        """Instantiate the Procgen environment"""
        # env setup
        envs = ProcgenEnv(
            num_envs=num_envs,
            env_name=env_id,
            start_level=start_level,
            num_levels=num_levels,
            distribution_mode=level_distribution,
        )
        envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space["rgb"]
        envs.is_vector_env = True
        envs = gym.wrappers.RecordEpisodeStatistics(envs)
        if capture_video:
            envs = ResizeRenderWrapper(envs, (256, 256))
            envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}", episode_trigger=lambda x: True)
        envs = gym.wrappers.NormalizeReward(envs, gamma=gamma)
        envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

        return envs
else:

    def make_procgen(
        env_id: str,
        num_envs: int,
        level_distribution: str,
        start_level: int,
        num_levels: int,
        capture_video: bool,
        gamma: float,
        run_name: str,
    ) -> ProcgenEnv:
        raise ModuleNotFoundError("gym package not found. Can't run ProcGen!")


def make_atari(env_id: str, idx: int, capture_video: bool, run_name: str) -> Callable[[], gymnasium.Env]:
    def thunk():
        if capture_video and idx == 0:
            env = gymnasium.make(env_id, render_mode="rgb_array")
            # Make a video of every episode
            env = gymnasium.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gymnasium.make(env_id)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gymnasium.wrappers.ResizeObservation(env, (84, 84))
        env = gymnasium.wrappers.GrayscaleObservation(env)
        env = gymnasium.wrappers.FrameStackObservation(env, 4)
        return env

    return thunk


def make_minigrid(
    env_id: str, dense_reward: bool, disable_orientation: bool, idx: int, capture_video: bool, run_name: str
) -> Callable[[], gymnasium.Env]:
    """Create a functions that create a GridWorlds."""

    def thunk():
        
        env = gymnasium.make(env_id, render_mode="rgb_array")

        # Default wrappers for MiniGrids
        # NOTE: This wrapper alters the step function, so it must be applied before observation wrappers
        if disable_orientation:
            env = NoOrientationActionWrapper(env)
        env = ImgObsWrapper(env)
        env = ChannelFirstObsWrapper(env)

        # Additional wrappers that are common to all MiniGrids
        if dense_reward:
            env = DenseRewardWrapper(env)
        if capture_video:
            env = gymnasium.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: True)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk
