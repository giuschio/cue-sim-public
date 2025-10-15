"""Interactive demo for CueSim environments via Gymnasium."""

from __future__ import annotations

import argparse
import time
from typing import Iterable

import gymnasium as gym
import numpy as np

import cuesim


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        choices=sorted(cuesim.DEFAULT_ENV_REGISTRY),
        default="Cuesim/ThreeBallHard-v0",
        help="Environment id to run (defaults to Cuesim/ThreeBallHard-v0).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of demo episodes to play.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed forwarded to the environment.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.1,
        help="Optional pause (seconds) between rendered steps.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    cuesim.register_gymnasium_environments()

    env_id = args.env
    options = cuesim.get_env_options(env_id)
    env = gym.make(
        env_id,
        options=options,
        headless=False,
        render_mode="human",
    )

    try:
        agent = cuesim.DirectShotAgent(env.unwrapped.env_options["physics"])
        for episode in range(1, args.episodes + 1):
            episode_seed = None if args.seed is None else args.seed + episode - 1
            observation, info = env.reset(seed=episode_seed)
            done = False
            cumulative_reward = 0.0
            while not done:
                action = np.asarray(agent.select_action(observation, None), dtype=np.float32)
                observation, reward, terminated, truncated, info = env.step(action)
                cumulative_reward += reward
                done = terminated or truncated
                if args.step_delay:
                    time.sleep(args.step_delay)
            print(f"Episode {episode}: reward={cumulative_reward:.3f}")
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
