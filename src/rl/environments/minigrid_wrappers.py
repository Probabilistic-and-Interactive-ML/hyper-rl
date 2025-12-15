from enum import IntEnum
from itertools import product

import gymnasium
import numpy as np
from minigrid.minigrid_env import MiniGridEnv


class SimpleActions(IntEnum):
    """Simple action space for MiniGrid environments."""

    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2


class NoOrientationActions(IntEnum):
    """Action space for MiniGrid environments where the agent can move in any direction."""

    # Move left, right, up, down
    left = 0
    right = 1
    up = 2
    down = 3


class ChannelFirstObsWrapper(gymnasium.core.ObservationWrapper):
    """Wrapper that transposes the state of Minigrid to have channel-first observations."""

    def __init__(self, env: MiniGridEnv):
        super().__init__(env)
        channel_first_shape = (
            env.observation_space.shape[2],
            env.observation_space.shape[0],
            env.observation_space.shape[1],
        )

        if "RGBImg" in str(env) and "FullyObsWrapper" not in str(env):
            high = 255
        else:
            high = 8

        self.observation_space = gymnasium.spaces.Box(low=0, high=high, shape=channel_first_shape, dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.transpose(observation, (2, 0, 1))


class TwoDEnvWrapper(gymnasium.core.ObservationWrapper):
    """A wrapper that returns the agent's (x, y) position as the observation.

    This wrapper requires the NoOrientationActionWrapper to be used, as the agent's orientation
    is not part of the observation and the actions are based on cardinal directions.
    """

    def __init__(self, env: MiniGridEnv):
        super().__init__(env)

        # Check that the NoOrientationActionWrapper is used with this wrapper.
        if "NoOrientationActionWrapper" not in str(env):
            raise ValueError("TwoDEnvWrapper requires NoOrientationActionWrapper.")

        # This wrapper is not compatible with the TabularObservationWrapper.
        if "TabularObservationWrapper" in str(env):
            raise ValueError("TwoDEnvWrapper is not compatible with TabularObservationWrapper.")

        # The observation space is a 2D position (x, y).
        self.observation_space = gymnasium.spaces.Box(
            low=np.zeros(2, dtype=np.int32),
            high=np.array([self.env.unwrapped.width - 1, self.env.unwrapped.height - 1], dtype=np.int32),
            shape=(2,),
            dtype=np.int32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        # Return the agent's x/y position.
        return self.env.unwrapped.agent_pos.astype(np.int32)


class TabularObservationWrapper(gymnasium.core.ObservationWrapper):
    """A wrapper that generates 2D states from a MiniGrid environment for tabular solvers."""

    def __init__(self, env: MiniGridEnv):
        super().__init__(env)
        # Define the observation space as a 2D position (x, y)
        self.observation_space = gymnasium.spaces.Box(
            low=np.zeros(3, np.int32),
            high=np.array([self.env.unwrapped.height - 1, self.env.unwrapped.width - 1, 3], np.int32),
            shape=(3,),
            dtype=np.int32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        # Extract the agent's x/y position from the environment
        agent_x, agent_y = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir
        # Concatenate with the agent's direction to form the state
        state = np.array([agent_x, agent_y, agent_dir], dtype=np.int32)
        return state


class NoOrientationActionWrapper(gymnasium.Wrapper):
    """A wrapper that changes the action space to move left, right, up, or down, irrespective of agent orientation."""

    def __init__(self, env: MiniGridEnv):
        super().__init__(env)
        self.actions = NoOrientationActions
        self.action_space = gymnasium.spaces.Discrete(len(self.actions))

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        env = self.env.unwrapped
        env.step_count += 1
        agent_pos = env.agent_pos

        reward = 0
        terminated = False
        truncated = False

        # Get the new position of the agent
        if action == self.actions.left:
            new_pos = agent_pos + np.array([-1, 0])
        elif action == self.actions.right:
            new_pos = agent_pos + np.array([1, 0])
        elif action == self.actions.up:
            new_pos = agent_pos + np.array([0, -1])
        elif action == self.actions.down:
            new_pos = agent_pos + np.array([0, 1])
        else:
            raise ValueError(f"Unknown action: {action}")

        # Check if the new position is valid
        fwd_cell = env.grid.get(*new_pos)
        if fwd_cell is None or fwd_cell.can_overlap():
            # Update the agent's position and keep the orientation fixed
            env.agent_pos = new_pos
            env.agent_dir = 0
        if fwd_cell is not None and fwd_cell.type == "goal":
            terminated = True
            reward = env._reward()
        if fwd_cell is not None and fwd_cell.type == "lava":
            terminated = True

        # We need to manually check for the goal condition after moving the agent
        if env.step_count >= env.max_steps:
            truncated = True
        info = {}

        # Re-render and generate new obs after moving into lava
        if self.render_mode == "human":
            env.render()
        obs = env.gen_obs()

        return obs, reward, terminated, truncated, info


class RemoveInvalidActionsWrapper(gymnasium.core.ActionWrapper):
    """A wrapper that removes invalid/unnecessary actions from the action space."""

    def __init__(self, env: MiniGridEnv):
        super().__init__(env)
        if "NoOrientationActionWrapper" in str(env):
            self.actions = NoOrientationActions
        else:
            self.actions = SimpleActions

        # Actions are discrete integer values
        self.action_space = gymnasium.spaces.Discrete(len(self.actions))

    def action(self, action: int) -> int:
        return action


class DenseRewardWrapper(gymnasium.core.RewardWrapper):
    """A wrapper that returns dense rewards for MiniGrid environments."""

    def __init__(self, env: MiniGridEnv):
        super().__init__(env)
        # Maximum distance between the agent and the goal excluding walls
        # Example: 9x9 grid has a max distance of 12 when agent is in (1,1) and goal in (7,7)
        self.max_dist = self.env.unwrapped.width + self.env.unwrapped.height - 6

        self.goal_position = None

    def reward(self, reward: float) -> float:
        # Get goal position if not already set
        if self.goal_position is None:
            # Get goal position in reverse, which should usually be faster
            for i, j in product(
                reversed(range(self.env.unwrapped.grid.width)), reversed(range(self.env.unwrapped.grid.height))
            ):
                if self.env.unwrapped.grid.get(i, j) is not None and self.env.unwrapped.grid.get(i, j).type == "goal":
                    self.goal_position = (i, j)
                    break

        agent_x, agent_y = self.env.unwrapped.agent_pos
        goal_x, goal_y = self.goal_position
        # Calculate the normalized Manhattan distance between the agent and the goal
        goal_distance = abs(agent_x - goal_x) + abs(agent_y - goal_y)
        normalized_distance = goal_distance / self.max_dist
        # Return the negative distance as the reward
        return reward - normalized_distance / self.env.unwrapped.max_steps


class LavaNegativeRewardWrapper(gymnasium.Wrapper):
    """Assigns a fixed negative reward when dying from lava instead of just terminating the episode."""

    def __init__(self, env: MiniGridEnv, lava_penalty: float = -0.1):
        super().__init__(env)
        self.lava_penalty = lava_penalty

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Step the env
        obs, reward, terminated, truncated, info = self.env.step(action)

        env = self.env.unwrapped
        # Get the contents of the cell in front of the agent
        curr_cell = env.grid.get(*env.agent_pos)

        fwd_pos = env.front_pos
        fwd_cell = env.grid.get(*fwd_pos)

        # Check if the agent is in lava and assign a negative reward
        if curr_cell is not None and curr_cell.type == "lava":
            reward = self.lava_penalty
        if terminated and action == env.actions.forward and fwd_cell is not None and fwd_cell.type == "lava":
            reward = self.lava_penalty

        return obs, reward, terminated, truncated, info


class LavaNotDeadWrapper(gymnasium.Wrapper):
    """Does not terminate the episode when the agent steps into the lava. Best used with LavaNegativeRewardWrapper."""

    def __init__(self, env: MiniGridEnv):
        super().__init__(env)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Step the env
        obs, reward, terminated, truncated, info = self.env.step(action)

        env = self.env.unwrapped
        # Get the contents of the cell in front of the agent
        curr_cell = env.grid.get(*env.agent_pos)

        fwd_pos = env.front_pos
        fwd_cell = env.grid.get(*fwd_pos)

        # Check if the agent is in lava/moving into lava and overwrite the terminated flag
        if curr_cell is not None and curr_cell.type == "lava":
            terminated = False
        if terminated and action == env.actions.forward and fwd_cell is not None and fwd_cell.type == "lava":
            terminated = False

            # Re-render and generate new obs after moving into lava
            if self.render_mode == "human":
                env.render()

            obs = env.gen_obs()

        return obs, reward, terminated, truncated, info
