# CueSim

CueSim provides fast billiards simulations that follow the Gymnasium API. It exposes configurable physics, environment presets for common training tasks, and helper utilities for inspection or heuristic agents.

## Features
- Gymnasium-compatible `BilliardsEnv` with vector action space and dense/sparse rewards
- Preset configurations for one-ball and multi-ball drills (including regulation-table physics)
- Rendering through Pygame for interactive play plus headless simulation for training
- Utility helpers for action search, plotting, augmentations, and heuristic shot controllers

## Installation

```bash
python -m pip install cuesim
```

To test the packaging workflow before publishing, build and upload to TestPyPI:

```bash
python -m pip install build twine
python -m build                   # creates dist/ artifacts
twine upload -r testpypi dist/*   # requires a TestPyPI account/token
```

Install from TestPyPI as:

```bash
python -m pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple cuesim
```

## Quick Start

```python
import gymnasium as gym
import cuesim

cuesim.register_gymnasium_environments()
env = gym.make("Cuesim/ThreeBallEasy-v0")

observation, info = env.reset(seed=42)
for _ in range(10):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
```

Instantiate a custom environment directly if you want to tweak the physics or rewards:

```python
from cuesim.environments import BilliardsEnv, get_env_options

options = get_env_options("ThreeBallRegulation-v0", {"physics.render_dt": 0.01})
env = BilliardsEnv(seed=0, options=options, headless=True)
```

## Included Environments

| Gym Id                         | Description                              |
|-------------------------------|------------------------------------------|
| `Cuesim/OneBall-v0`           | Pocket a single target ball in one shot. |
| `Cuesim/ThreeBallEasy-v0`     | Three-ball drill on the practice table.  |
| `Cuesim/ThreeBallRegulation-v0` | Three-ball drill with regulation physics. |
| `Cuesim/ThreeBallRegulationSparse-v0` | Sparse reward variant of the regulation setup. |

Use `cuesim.DEFAULT_ENV_REGISTRY` to list the bundled Gym ids.
Pass an optional `prefix` to `cuesim.register_gymnasium_environments` if you also want namespaced ids (e.g. `Cuesim/ThreeBallEasy-v0`).

## Development
- Format and lint with `ruff`/`black` (see optional dependencies).
- Run smoke tests: `pytest tests/` (add your own tests around reset/step loops).
- For local installs, use editable mode: `python -m pip install -e .`.

## License

The license for CueSim matches the root project. Update `pyproject.toml` before releasing to PyPI if you change or formalize the licensing terms.
