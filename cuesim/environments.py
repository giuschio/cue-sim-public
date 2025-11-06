import copy
from functools import reduce

import gymnasium as gym
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
from numpy.random._generator import Generator as Generator

from cuesim.simulators import BilliardsSimulator
from cuesim.utils import (
    random_2d,
    sort_vectors,
    uniform_2d_numpy,
    vec2deg,
    deg2vec,
    rotate_vec,
)
from cuesim.utils import angle as angle_deg

DEFAULT_ENV_ID = "Cuesim/ThreeBallHard-v0"


class Donut(spaces.Space):
    def __init__(self, low=1.0, high=1.0, shape=(2,), dtype=np.float32):
        super().__init__(shape=shape, dtype=dtype, seed=None)
        # restrict the action space
        assert (low == high) or (low == 0)
        self.high = high
        self.low = low
        self.max_action = high
        self.min_action = low
        # tollerance = 0.1% of high
        self._toll = 0.001 * high

    def sample(self, mask=None):
        x, y = random_2d(self._np_random, min_radius=self.low, max_radius=self.high)
        return np.array([x, y], dtype=np.float32)

    def contains(self, action):
        if action is None:
            return False
        arr = np.asarray(action, dtype=np.float32)
        if arr.shape != self.shape:
            return False
        norm = np.linalg.norm(arr)
        return bool(self.low - self._toll <= norm <= self.high + self._toll)


__physics_options = dict(
    easy=dict(
        damping=0.7,
        table_width=0.8,
        table_height=0.4,
        holes_width=0.05,  # was 12 cm
        ball_radius=0.02,
        friction=0.5,
        position_jitter=0.002,
        velocity_jitter=0.002,
        velocity_bias=0.0,
        state_bias_x=0.0,
        state_bias_y=0.0,
        dt=0.04,
        render_dt=0.02,
        side_pockets=False,
        gravity_bias_x=0.,
        gravity_bias_y=0.
    ),
    regulation=dict(
        damping=0.7,
        table_width=0.88,
        table_height=0.44,
        holes_width=0.025,  # was 12 cm
        ball_radius=0.01125,
        friction=0.5,
        position_jitter=0.001,
        velocity_jitter=0.0,
        velocity_bias=0.0,
        state_bias_x=0.0,
        state_bias_y=0.0,
        dt=0.02,
        render_dt=0.02,
        side_pockets=True,
        gravity_bias_x=0.,
        gravity_bias_y=0.
    ),
)

__env_options = {
    "Cuesim/OneBall-v0": dict(
        n_target_balls=1,
        n_penalty_balls=0,
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        normalize_actions=True,
        physics=__physics_options["easy"],
        pocket_observations=False,
        max_steps=10,
        train_reward="n_pocketed",
        eval_reward="n_pocketed",
    ),
    "Cuesim/ThreeBallEasy-v0": dict(
        n_target_balls=3,
        n_penalty_balls=0,
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        normalize_actions=True,
        physics=__physics_options["easy"],
        pocket_observations=False,
        max_steps=10,
        train_reward="n_pocketed",
        eval_reward="n_pocketed",
    ),
    "Cuesim/ThreeBallHard-v0": dict(
        n_target_balls=3,
        n_penalty_balls=0,
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        normalize_actions=True,
        physics=__physics_options["regulation"],
        pocket_observations=False,
        max_steps=10,
        train_reward="n_pocketed",
        eval_reward="n_pocketed",
    ),
    "Cuesim/ThreeBallHard-Cuelearner": dict(
        n_target_balls=3,
        n_penalty_balls=0,
        action_space=Donut(low=1.0, high=1.0),
        normalize_actions=True,
        physics=__physics_options["regulation"],
        pocket_observations=False,
        max_steps=10,
        train_reward="n_pocketed",
        eval_reward="n_pocketed",
    ),
    # "Cuesim/ThreeBallRegulationSparse-v0": dict(
    #     n_target_balls=3,
    #     n_penalty_balls=0,
    #     action_space=Donut(low=1.0, high=1.0),
    #     normalize_actions=True,
    #     physics=__physics_options["regulation"],
    #     pocket_observations=False,
    #     max_steps=10,
    #     train_reward="n_steps",
    #     eval_reward="n_steps",
    # ),
}

DEFAULT_ENV_REGISTRY = tuple(__env_options.keys())


def update_dict(d, update):
    for key, value in update.items():
        path = key.split(".")
        reduce(lambda d, k: d[k], path[:-1], d)[path[-1]] = value
    return d


def get_env_options(env_id, env_options=None):
    if env_id not in __env_options:
        raise KeyError(f"Unknown environment id {env_id!r}.")
    options = copy.deepcopy(__env_options[env_id])
    if env_options is not None:
        options = update_dict(options, env_options)
    return options


def n_pocketed(init_state, current_state):
    n_pocketed = current_state["num_pocketed"] - init_state["num_pocketed"]
    return n_pocketed - (1 if current_state["state"]["cue_ball"]["pocketed"] else 0)

def n_steps(init_state, current_state):
    if current_state["done"]:
        return 1.0 - (current_state["step_count"]/current_state["max_steps"])
    else:
        return 0.0


def get_reward_function(rew_key):
    __task_rewards = dict(
        n_pocketed=n_pocketed,
        n_steps=n_steps)
    return __task_rewards[rew_key]


__n_discrete_bins = 360
BinnedActionSpace = spaces.Discrete(__n_discrete_bins)


def get_row(num_balls, rad_balls, x_row):
    d = rad_balls * 2
    y_pos = np.array([i * d for i in range(num_balls)])
    y_pos -= ((num_balls - 1.0) * d) / 2

    positions = [[x_row, y] for y in y_pos]
    return positions


def get_triangle(num_balls, rad_balls, x_head):
    positions = []
    balls_to_assign = num_balls
    row_index = 0
    while balls_to_assign > 0:
        positions += get_row(
            min(row_index + 1, balls_to_assign),
            rad_balls,
            x_head + rad_balls * row_index * 2,
        )
        row_index += 1
        balls_to_assign = num_balls - len(positions)
    return positions


class BilliardsEnv(gym.Env):
    """Gymnasium-compatible billiards environment."""

    metadata = {"render_modes": ["human"], "render_fps": 25}

    def __init__(
        self,
        seed=None,
        options=None,
        render_mode=None,
        headless=None,
        **kwargs,
    ):
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render mode {render_mode!r}. "
                f"Available modes: {self.metadata['render_modes']}"
            )

        if options is None:
            options = copy.deepcopy(__env_options[DEFAULT_ENV_ID])
        else:
            options = copy.deepcopy(options)

        if headless is None:
            headless = render_mode != "human"
        if render_mode is None and not headless:
            render_mode = "human"

        self.render_mode = render_mode
        self.headless = bool(headless)

        if not self.headless:
            pygame.init()
            pygame.display.set_caption("Billiards Simulator")

        self._options_template = copy.deepcopy(options)
        sim_options = copy.deepcopy(options)
        self.step_count = 0
        self.sim = BilliardsSimulator(seed=seed, options=sim_options)
        self.__options = self.sim.options
        self.__rng = np.random.default_rng(seed)
        self._base_seed = seed
        self._update_render_metadata()

        # Define action space and observation space
        self.action_space = self.__options["action_space"]
        self.ball_observation_dimension = 3
        obs_dimension = (
            int(self.sim.num_balls * self.ball_observation_dimension + 8)
            if options["pocket_observations"]
            else int(self.sim.num_balls * self.ball_observation_dimension)
        )

        # Observation bounds can vary by component (positions, flags, pockets), so we
        # keep them unbounded to avoid mismatches with the simulator outputs.
        low = np.full((obs_dimension,), -np.inf, dtype=np.float32)
        high = np.full((obs_dimension,), np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # seed the action space
        self.action_space.seed(seed=seed)

        # define the reward function
        self.__compute_train_reward = get_reward_function(options["train_reward"])
        self.__compute_eval_reward = get_reward_function(options["eval_reward"])
        self.cumulative_reward = 0.0
        self.spec = EnvSpec(
            id="Cuesim-Billiards-v0", max_episode_steps=self.__options["max_steps"]
        )

    @property
    def env_info(self):
        env_info = {
            "state_dim": self.observation_space.shape[0],
            "action_dim": self.action_space.shape[0],
            "min_action": getattr(
                self.action_space, "min_action", np.asarray(self.action_space.low)
            ),
            "max_action": getattr(
                self.action_space, "max_action", np.asarray(self.action_space.high)
            ),
            "action_set": uniform_2d_numpy(interval=0.1),
        }
        return env_info

    @property
    def env_options(self):
        return self.__options

    def _update_render_metadata(self):
        render_dt = self.__options["physics"].get("render_dt", 0.04)
        if render_dt > 0:
            self.metadata["render_fps"] = int(round(1 / render_dt))

    def __discrete_to_action(bin_idx):
        # convert bins to degrees
        angle = np.deg2rad(360.0 / __n_discrete_bins * bin_idx)
        x = np.cos(angle)
        y = np.sin(angle)
        return np.array([x, y])

    def __ball_observation(self, ball_info):
        # each ball is (x,y,pocketed)
        pocketed = 0.5 if ball_info["pocketed"] else -0.5
        obs = [ball_info["position"][0], ball_info["position"][1], pocketed]
        return np.array(obs)

    def __compute_observation(self, info):
        table_np = info["pocket_positions"]

        # guarantee ordering of the state vector, even if it should be
        # the case regardless
        s_cue = [self.__ball_observation(info["state"]["cue_ball"])]
        s_target = [
            self.__ball_observation(v)
            for k, v in info["state"].items()
            if "target" in k
        ]
        s_penalty = [
            self.__ball_observation(v)
            for k, v in info["state"].items()
            if "penalty" in k
        ]
        sv = sort_vectors

        state = s_cue + sv(s_target) + sv(s_penalty)
        state_np = np.array(state)
        state_np[:, 0] += self.__options["physics"]["state_bias_x"]
        state_np[:, 1] += self.__options["physics"]["state_bias_y"]
        if self.__options["pocket_observations"]:
            return np.concatenate(
                (state_np.flatten(), table_np.flatten())
            ).astype(np.float32, copy=False)
        else:
            return state_np.flatten().astype(np.float32, copy=False)

    def __detect_foul(self, info):
        return info["state"]["cue_ball"]["pocketed"]

    def __shuffle_cue_ball(self, info, toll):
        # positions of other balls
        positions = np.array(
            [v["position"] for k, v in info["state"].items() if "cue" not in k]
        )
        valid = False
        while not valid:
            x = self.__rng.uniform(-self.sim.w / 2, -self.sim.w / 4)
            y = self.__rng.uniform(-self.sim.h / 2, self.sim.h / 2)

            cue_position = np.array([x, y])
            diff = positions - cue_position
            valid = np.min(np.sqrt(np.sum(diff**2, axis=1))) > toll

        self.sim.set_position(
            "cue_ball",
            cue_position,
            jitter=0.0,
        )

    def __step_util(self):
        done = False

        next_render_time = self.sim.time
        while not done:
            done = self.sim.step()
            if (
                not self.headless
                and self.render_mode == "human"
                and self.sim.time >= next_render_time
            ):
                next_render_time = (
                    next_render_time + self.__options["physics"]["render_dt"]
                )
                self.sim.draw()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None

        [self.sim.set_velocity(k, [0, 0], 0) for k in self.sim.balls]
        [self.sim.set_angular_velocity(k, 0, 0) for k in self.sim.balls]
        current_state = self.sim.get_info()
        return current_state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.__rng = np.random.default_rng(seed)
            sim_options = copy.deepcopy(self._options_template)
            self.sim = BilliardsSimulator(seed=seed, options=sim_options)
            self.__options = self.sim.options
            self.action_space = self.__options["action_space"]
            self.action_space.seed(seed=seed)
            self._update_render_metadata()
        self.sim.reset_clock()
        self.step_count = 0
        self.cumulative_reward = 0
        head_spot = [-self.sim.w / 4, 0.0]
        foot_spot = [self.sim.w / 4, 0.0]

        self.sim.set_position("cue_ball", head_spot)
        ball_positions = get_triangle(
            self.sim.n_target_balls,
            self.__options["physics"]["ball_radius"] * 1.05,
            x_head=foot_spot[0],
        )
        for idx in range(self.sim.n_target_balls):
            self.sim.set_position(
                f"target_ball_{idx+1}",
                ball_positions[idx],
                jitter=self.__options["physics"]["position_jitter"],
            )

        [self.sim.set_velocity(k, [0, 0], 0) for k in self.sim.balls]
        [self.sim.set_angular_velocity(k, 0, 0) for k in self.sim.balls]

        self.__step_util()
        info = self.sim.get_info()
        self.sim.set_text(f"STEPS: 1/{self.__options['max_steps']} -- TOT. REWARD: 0")
        info["step_count"] = self.step_count
        info["cumulative_reward"] = self.cumulative_reward
        return self.__compute_observation(info), info

    def reset_to(self, info):
        self.sim.reset_clock(info["time"])
        self.step_count = info["step_count"]
        self.cumulative_reward = info["cumulative_reward"]
        for key in info["state"].keys():
            self.sim.set_position(key, info["state"][key]["position"])
            self.sim.set_velocity(key, info["state"][key]["velocity"])
            self.sim.set_angular_velocity(key, info["state"][key]["angular_velocity"])
        self.sim.set_text(
            f"STEPS: {self.step_count+1}/{self.__options['max_steps']} -- TOT. REWARD: {int(self.cumulative_reward)}"
        )
        return self.__compute_observation(info), info

    def step(self, action):
        # action is a 2D vector of x and y velocities
        if type(self.action_space) is spaces.Discrete:
            action = self.__discrete_to_action(action)
        action = np.asarray(action, dtype=np.float32)

        if self.__options["normalize_actions"]:
            norm = np.linalg.norm(action)
            if norm > 0:
                action = action / norm

        if abs(self.__options["physics"]["velocity_bias"]) > 0:
            action = rotate_vec(action, self.__options["physics"]["velocity_bias"])
            action = np.asarray(action, dtype=np.float32)

        vel = action * 0.5
        # increase the step counter
        self.step_count += 1
        init_info = self.sim.get_info()

        self.sim.set_velocity(
            "cue_ball", vel, jitter=self.__options["physics"]["velocity_jitter"]
        )

        current_info = self.__step_util()
        if self.__detect_foul(current_info):
            # reset the ball positions
            for ball_name, ball_state in init_info["state"].items():
                self.sim.set_position(ball_name, ball_state["position"])
            # shuffle the cue ball
            self.__shuffle_cue_ball(
                current_info, toll=self.__options["physics"]["ball_radius"] * 2.0
            )
            current_info = self.sim.get_info()

        terminated = bool(current_info["num_target_pocketed"] == self.sim.n_target_balls)
        truncated = bool(self.step_count >= self.__options["max_steps"])
        done = terminated or truncated
        current_info["terminated"] = terminated
        current_info["truncated"] = truncated
        current_info["done"] = done
        current_info["step_count"] = self.step_count
        current_info["max_steps"] = self.__options["max_steps"]
        train_reward = self.__compute_train_reward(
            init_state=init_info, current_state=current_info
        )
        observation = self.__compute_observation(current_info)

        eval_reward = self.__compute_eval_reward(
            init_state=init_info, current_state=current_info
        )
        current_info["train_reward"] = train_reward
        current_info["eval_reward"] = eval_reward
        self.cumulative_reward += eval_reward
        current_info["cumulative_reward"] = self.cumulative_reward

        self.sim.set_text(
            f"STEPS: {self.step_count+1}/{self.__options['max_steps']} -- TOT. REWARD: {int(self.cumulative_reward)}"
        )

        return (
            observation,
            train_reward,
            terminated,
            truncated,
            current_info,
        )  # Return the next observation, reward, done, truncated, info

    def get_best_action(self):
        # sample the action space exhaustively and then choose
        # the action with the highest reward
        best_reward = -np.inf
        best_action = None

        info = self.sim.get_info()
        info["step_count"] = self.step_count
        info["cumulative_reward"] = self.cumulative_reward

        action_set = self.env_info["action_set"]
        for action in action_set:
            next_state, reward, done, truncated, next_info = self.step(action)

            if reward > best_reward:
                best_reward = reward
                best_action = action

            self.reset_to(info)

        return best_action

    def get_closest_greedy_action(self, action, search_width=180.0):
        # given a root action, find the closest action with a reward == 1
        best_reward = -np.inf
        best_action = action

        info = self.sim.get_info()
        info["step_count"] = self.step_count
        info["cumulative_reward"] = self.cumulative_reward

        angle_step = 0.1
        half_iterations = int((search_width / angle_step) + 0.5)
        max_iterations = 1 + half_iterations * 2

        if (
            self.step_count == 0
            and angle_deg(action, np.array([1.0, 0])) < search_width
        ):
            return np.array([1.0, 0])

        angles_deg = np.arange(0, 360, angle_step)
        indices = np.empty((angles_deg.size,), dtype=int)
        indices[0::2] = np.arange(0, (angles_deg.size + 1) // 2)
        indices[1::2] = np.arange(angles_deg.size - 1, angles_deg.size // 2 - 1, -1)
        angles_deg = angles_deg[indices]

        action = vec2deg(action)
        angles_deg = angles_deg[:max_iterations]
        best_action = None

        for angle in angles_deg:
            proposed_action = deg2vec(action + angle)
            next_state, reward, done, truncated, next_info = self.step(proposed_action)
            if reward == 1:
                best_reward = reward
                best_action = proposed_action
                break

            self.reset_to(info)
        self.reset_to(info)

        return best_action

    # def get_closest_robust_action(self, action, search_width=180.0):
    #     # given a root action, find the closest action with a reward == 1
    #     best_reward = -np.inf
    #     best_action = action

    #     info = self.sim.get_info()
    #     info["step_count"] = self.step_count
    #     info["cumulative_reward"] = self.cumulative_reward

    #     angle_step = 0.1
    #     half_iterations = int((search_width / angle_step) + 0.5)
    #     max_iterations = 1 + half_iterations * 2

    #     if (
    #         self.step_count == 0
    #         and angle_deg(action, np.array([1.0, 0])) < search_width
    #     ):
    #         return np.array([1.0, 0])

    #     angles_deg = np.arange(0, 360, angle_step)
    #     indices = np.empty((angles_deg.size,), dtype=int)
    #     indices[0::2] = np.arange(0, (angles_deg.size + 1) // 2)
    #     indices[1::2] = np.arange(angles_deg.size - 1, angles_deg.size // 2 - 1, -1)
    #     angles_deg = angles_deg[indices]

    #     action = vec2deg(action)
    #     angles_deg = angles_deg[:max_iterations]
    #     best_action = None

    #     n = 1  # Number of actions to check before and after the current action
    #     rewards_buffer = []
    #     actions_buffer = []

    #     # Loop through each angle in the specified sequence
    #     for i, angle in enumerate(angles_deg):
    #         proposed_action_angle = (action + angle) % 360.0
    #         proposed_action = deg2vec(proposed_action_angle)
    #         next_state, reward, done, truncated, next_info = self.step(proposed_action)

    #         # Store the recent actions and their rewards
    #         rewards_buffer.append(reward)
    #         actions_buffer.append(proposed_action)

    #         # We need at least 2*n + 1 actions to start checking the grouping condition
    #         if len(rewards_buffer) >= 2 * n + 1:
    #             # Create a list of indices to check based on the pattern of angles_deg
    #             # For example, at index 4 (angle = 4), we need to check indices [0, 2, 4]
    #             indices_to_check = [i - 2 * k for k in range(n, -1, -1)] + [
    #                 i + 2 * k for k in range(1, n + 1)
    #             ]

    #             # Filter indices to keep only valid ones
    #             indices_to_check = [
    #                 idx for idx in indices_to_check if 0 <= idx < len(rewards_buffer)
    #             ]

    #             # Check the rewards of the selected indices
    #             if all(rewards_buffer[idx] == 1 for idx in indices_to_check):
    #                 best_reward = rewards_buffer[i]  # Reward of the current action
    #                 best_action = actions_buffer[i]
    #                 break

    #         # Reset the environment to the previous state after each step
    #         self.reset_to(info)
    #     self.reset_to(info)

    #     return best_action

    def plot_init(self):
        fig, ax = plt.subplots()
        ax.set_xlim(-1.5 * self.sim.w / 2, 1.5 * self.sim.w / 2)
        ax.set_ylim(-1.5 * self.sim.h / 2, 1.5 * self.sim.h / 2)
        ax.set_aspect("equal", adjustable="box")
        ax.add_patch(
            plt.Rectangle(
                (-self.sim.w / 2, -self.sim.h / 2), self.sim.w, self.sim.h, fill=False
            )
        )

        state = self.sim.get_info()["state"]
        colors = {"cue": "khaki", "target": "red", "penalty": "black"}
        for ball_name, ball_state in state.items():
            color = colors.get(
                ball_name.split("_")[0], "black"
            )  # Default to black if no match
            ax.add_patch(
                plt.Circle(
                    (ball_state["position"][0], ball_state["position"][1]),
                    self.sim.options["ball_radius"],
                    color=color,
                )
            )

        self.fig, self.ax = fig, ax
        plt.gca().invert_yaxis()
        return fig, ax

    def plot_segment(self, start, direction, length, color):
        direction = direction / (np.linalg.norm(direction) + 1e-5)
        end_pos = start + direction * length
        self.ax.plot([start[0], end_pos[0]], [start[1], end_pos[1]], color=color)

    def plot_text(self, text, font_size=32):
        for tfield in self.ax.texts:
            tfield.remove()
        self.ax.text(
            -1.5 * self.sim.w / 2,
            -1.5 * self.sim.h / 2,
            text,
            fontsize=font_size,
            color="black",
            ha="left",
        )

    def plot_actions_rewards(self, actions, rewards):
        norm = mcolors.Normalize(vmin=-1, vmax=1)
        mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet)
        state = self.sim.get_info()["state"]
        cue_ball_pos = state["cue_ball"]["position"]
        for action, reward in zip(actions, rewards):
            scaled_action = 0.05 * action
            self.plot_segment(
                start=cue_ball_pos + scaled_action,
                direction=scaled_action,
                length=(1 + reward) * 0.05,
                color=mapper.to_rgba(np.clip(reward, -1, 1)),
            )
        plt.colorbar(mapper, ax=self.ax, label="Expected Reward")
        self.plot_text(f"Max est Q: {max(rewards):.2f}")

    def plot_action(self, action, color):
        state = self.sim.get_info()["state"]
        cue_ball_pos = state["cue_ball"]["position"]
        self.plot_segment(start=cue_ball_pos, direction=action, length=0.1, color=color)

    def plot_fullscreen(self):
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())

    def render(self):
        if self.render_mode == "human":
            self.sim.draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    break
            return None
        raise NotImplementedError("Render mode not supported.")

    def close(self):
        if not self.headless and pygame.get_init():
            pygame.display.quit()
            pygame.quit()
        if getattr(self, "fig", None) is not None:
            plt.close(self.fig)


def get_envs(seed, options, n=2):
    # Get n independent environment instances
    envs = [
        BilliardsEnv(
            seed=seed + 123 * i if seed is not None else None,
            options=copy.deepcopy(options),
            headless=True,
        )
        for i in range(n)
    ]
    return envs
