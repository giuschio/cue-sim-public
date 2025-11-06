#!/usr/bin/env python3
"""Render a trained SAC agent on the CueSim ThreeBallEasy environment."""

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC

import cuesim

ENV_ID = "Cuesim/ThreeBallEasy-v0"
MODEL_PATH = Path("runs/simple_sac/threeball_easy_sac.zip")
EPISODES = 3
SEED = 0
DEVICE = "auto"
DETERMINISTIC = True


def main() -> None:
    cuesim.register_gymnasium_environments()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    env_options = cuesim.get_env_options(ENV_ID)
    env = gym.make(ENV_ID, options=env_options, headless=False, render_mode="human")
    env.action_space.seed(SEED)

    model = SAC.load(str(MODEL_PATH), env=env, device=DEVICE)

    try:
        for episode in range(1, EPISODES + 1):
            observation, _ = env.reset(seed=SEED + episode - 1)
            done = False
            cumulative_reward = 0.0
            steps = 0
            while not done:
                action, _ = model.predict(observation, deterministic=DETERMINISTIC)
                observation, reward, terminated, truncated, info = env.step(action)
                cumulative_reward += float(reward)
                done = terminated or truncated
                steps = info.get("step_count", steps + 1)
            print(f"Episode {episode}: reward={cumulative_reward:.3f}, steps={steps}")
    except KeyboardInterrupt:
        print("\nRendering interrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
