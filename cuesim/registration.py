"""Helpers for registering CueSim environments with Gymnasium."""

from __future__ import annotations


from gymnasium.envs.registration import register
from gymnasium.error import Error as GymError

from .environments import DEFAULT_ENV_REGISTRY, get_env_options




def register_gymnasium_environments() -> None:
    """Register all default CueSim environments with Gymnasium.
    """

    for env_id in DEFAULT_ENV_REGISTRY:
        candidate_ids = [env_id]

        for gym_id in candidate_ids:
            options = get_env_options(env_id)
            try:
                register(
                    id=gym_id,
                    entry_point="cuesim.environments:BilliardsEnv",
                    kwargs={"options": options, "headless": True},
                    max_episode_steps=options["max_steps"],
                )
            except GymError as exc:  # pragma: no cover - gym raises various subclasses
                # Ignore re-registration attempts while still surfacing unexpected errors.
                if "exists" not in str(exc).lower():
                    raise


__all__ = ["DEFAULT_ENV_REGISTRY", "register_gymnasium_environments"]
