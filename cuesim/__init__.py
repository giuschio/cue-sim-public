"""CueSim: Gymnasium-compatible billiards simulation environments."""

from .augmentations import augment, mirror_x, mirror_y
from .environments import (
    BilliardsEnv,
    BinnedActionSpace,
    Donut,
    get_env_options,
    get_envs,
)
from .heuristic_agents import DirectShotAgent, AimedShotAgent, get_heuristic
from .registration import (
    DEFAULT_ENV_REGISTRY,
    register_gymnasium_environments,
)
from .simulators import BilliardsSimulator
from .utils import (
    angle,
    deg2vec,
    random_2d,
    rotate_vec,
    sort_vectors,
    uniform_2d_numpy,
    vec2deg,
)

__all__ = [
    "augment",
    "mirror_x",
    "mirror_y",
    "BilliardsEnv",
    "BinnedActionSpace",
    "Donut",
    "get_env_options",
    "get_envs",
    "DirectShotAgent",
    "AimedShotAgent",
    "get_heuristic",
    "register_gymnasium_environments",
    "DEFAULT_ENV_REGISTRY",
    "BilliardsSimulator",
    "angle",
    "deg2vec",
    "random_2d",
    "rotate_vec",
    "sort_vectors",
    "uniform_2d_numpy",
    "vec2deg",
]

__version__ = "0.1.0"
