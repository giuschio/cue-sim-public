#!/usr/bin/env python3
"""Minimal SAC training loop for the CueSim ThreeBallEasy task."""

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC

import cuesim

ENV_ID = "Cuesim/ThreeBallEasy-v0"
TOTAL_TIMESTEPS = 200_000
SEED = 0
LEARNING_RATE = 3e-4
DEVICE = "auto"
LOG_DIR = Path("runs/simple_sac")
MODEL_NAME = "threeball_easy_sac"


def main() -> None:
    cuesim.register_gymnasium_environments()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env_options = cuesim.get_env_options(ENV_ID)
    env = gym.make(ENV_ID, options=env_options, headless=True, render_mode=None)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        verbose=1,
        seed=SEED,
        device=DEVICE,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    save_path = LOG_DIR / MODEL_NAME
    model.save(str(save_path))
    print(f"Saved model to {save_path.with_suffix('.zip')}")

    env.close()


if __name__ == "__main__":
    main()
