import cv2
import gym
import numpy as np


class ResizeRenderWrapper(gym.Wrapper):
    """
    A wrapper to resize the output of env.render(mode='rgb_array').
    """

    def __init__(self, env: gym.Env, size: tuple[int, int]):
        """
        Args:
            env: The environment to wrap.
            size: The desired output size in (width, height).
        """
        super().__init__(env)
        self.size = size
        # Adjust the observation space if the wrapper is also used for observations
        # For this use case (video recording), it's not strictly necessary.
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(size[1], size[0], 3), dtype=np.uint8)

    def render(self, mode: str = "human", **kwargs) -> np.ndarray | None:
        """
        Renders the environment and resizes the output frame.
        """
        if mode == "rgb_array":
            frame = self.env.render(mode="rgb_array", **kwargs)
            if frame is not None:
                # Upscale the frame to the desired size.
                # cv2.INTER_CUBIC is a good choice for upscaling to avoid blockiness.
                return cv2.resize(frame, self.size, interpolation=cv2.INTER_CUBIC)
            return None
        else:
            # Fallback to the original render method for other modes.
            return self.env.render(mode=mode, **kwargs)
