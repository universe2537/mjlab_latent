![Project banner](https://raw.githubusercontent.com/mujocolab/mjlab/main/docs/source/_static/mjlab-banner.jpg)

# mjlab

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/mujocolab/mjlab/ci.yml?branch=main)](https://github.com/mujocolab/mjlab/actions/workflows/ci.yml?query=branch%3Amain)
[![Documentation](https://github.com/mujocolab/mjlab/actions/workflows/docs.yml/badge.svg)](https://mujocolab.github.io/mjlab/)
[![License](https://img.shields.io/github/license/mujocolab/mjlab)](https://github.com/mujocolab/mjlab/blob/main/LICENSE)
[![Nightly Benchmarks](https://img.shields.io/badge/Nightly-Benchmarks-blue)](https://mujocolab.github.io/mjlab/nightly/)
[![PyPI](https://img.shields.io/pypi/v/mjlab)](https://pypi.org/project/mjlab/)
[![PyPI downloads](https://img.shields.io/pypi/dm/mjlab?color=blue)](https://pypistats.org/packages/mjlab)

mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s manager-based API with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), a GPU-accelerated version of [MuJoCo](https://github.com/google-deepmind/mujoco).
The framework provides composable building blocks for environment design,
with minimal dependencies and direct access to native MuJoCo data structures.

## Getting Started

mjlab requires an NVIDIA GPU for training. macOS is supported for evaluation only.

**Try it now:**

Run the demo (no installation needed):

```bash
uvx --from mjlab --refresh demo
```

Or try in [Google Colab](https://colab.research.google.com/github/mujocolab/mjlab/blob/main/notebooks/demo.ipynb) (no local setup required).

**Install from source:**

```bash
git clone https://github.com/mujocolab/mjlab.git && cd mjlab
uv run demo
```

For alternative installation methods (PyPI, Docker), see the [Installation Guide](https://mujocolab.github.io/mjlab/main/source/installation.html).

## Training Examples

### 1. Velocity Tracking

Train a Unitree G1 humanoid to follow velocity commands on flat terrain:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096
```

**Multi-GPU Training:** Scale to multiple GPUs using `--gpu-ids`:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --gpu-ids "[0, 1]" \
  --env.scene.num-envs 4096
```

See the [Distributed Training guide](https://mujocolab.github.io/mjlab/main/source/training/distributed_training.html) for details.

Evaluate a policy while training (fetches latest checkpoint from Weights & Biases):

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 2. Motion Imitation

Train a humanoid to mimic reference motions. See the [motion imitation guide](https://mujocolab.github.io/mjlab/main/source/training/motion_imitation.html) for preprocessing setup.

```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name your-org/motions/motion-name --env.scene.num-envs 4096
uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 3. Sanity-check with Dummy Agents

Use built-in agents to sanity check your MDP before training:

```bash
uv run play Mjlab-Your-Task-Id --agent zero  # Sends zero actions
uv run play Mjlab-Your-Task-Id --agent random  # Sends uniform random actions
```

When running motion-tracking tasks, add `--registry-name your-org/motions/motion-name` to the command.


## Documentation

Full documentation is available at **[mujocolab.github.io/mjlab](https://mujocolab.github.io/mjlab/)**.

## Repository Layout

The repository follows a fairly standard Python project layout: source code lives in
`src/mjlab`, tests live in `tests`, documentation lives in `docs`, and helper scripts
for development and automation live alongside them at the repository root.

### Top-level directories

| Path | Purpose |
|------|---------|
| `src/` | Python source tree. The installable package itself lives in `src/mjlab/`. |
| `tests/` | Pytest suites covering package modules, task configs, viewers, notebooks, and smoke tests. |
| `docs/` | Sphinx documentation source, templates, and site configuration for the published docs. |
| `scripts/` | Repository-level helper scripts for demos, benchmarks, cloud workflows, conversion, and Docker utilities. |
| `notebooks/` | Jupyter and Colab notebooks for quickstarts, demos, and interactive exploration. |
| `data/` | Local data assets used by examples and task pipelines, such as motion or experiment inputs. |
| `artifacts/` | Generated outputs such as exported scenes or other local build artifacts. |
| `logs/` | Training and evaluation logs written by local runs. |
| `wandb/` | Weights & Biases run metadata and cached experiment outputs. |
| `typings/` | Local type stubs and typing helpers used by static analysis tools. |

Other notable top-level files include `pyproject.toml` for packaging, dependencies,
and CLI entry points, `Makefile` for common development commands, and
`CONTRIBUTING.md` / `RELEASING.md` for contributor workflows.

### `src/mjlab/` package layout

| Path | Purpose |
|------|---------|
| `actuator/` | Actuator abstractions and implementations, including MuJoCo built-in actuators, XML-backed actuators, PD/DC actuators, and learned actuator models. |
| `asset_zoo/` | Bundled robot assets and MJCF-based robot definitions, plus per-robot constants and configuration helpers. |
| `entity/` | Core entity abstraction that loads and edits `MjSpec`s, applies articulation/collision config, and exposes structured access to bodies, joints, sites, and actuators. |
| `envs/` | Manager-based RL environment definitions, shared environment types, and MDP-related utilities used to build tasks. |
| `managers/` | Modular managers for actions, observations, rewards, terminations, commands, curriculum, events, metrics, and recording. |
| `rl/` | RSL-RL integration layer, runners, exporters, and vector-environment wrappers used by training scripts. |
| `scene/` | Scene assembly utilities that combine terrains, entities, and sensors into batched MuJoCo scenes and export them when needed. |
| `sensor/` | Sensor framework and sensor implementations such as camera, contact, raycast, built-in MuJoCo, and terrain-height sensors. |
| `sim/` | Low-level simulation configuration and runtime bridge to MuJoCo Warp, including randomization and mesh-variant support. |
| `tasks/` | Task registry and built-in task families such as velocity tracking, motion tracking, manipulation, and cartpole. |
| `terrains/` | Procedural terrain generators, terrain entities, and related configuration for flat, primitive, and heightfield terrains. |
| `utils/` | Shared utilities for spec editing, XML handling, logging, randomness, GPU selection, wrappers, noise, and other cross-cutting helpers. |
| `viewer/` | Visualization and rendering stack, including native viewers, offscreen rendering, debug visualization, and Viser integration. |
| `scripts/` | Package-level CLI entry points such as `train`, `play`, `demo`, `list-envs`, `export-scene`, and diagnostic tools. |

## Development

```bash
make test          # Run all tests
make test-fast     # Skip slow tests
make format        # Format and lint
make docs          # Build docs locally
```

For development setup: `uvx pre-commit install`

## Citation

mjlab is used in published research and open-source robotics projects. See the [Research](https://mujocolab.github.io/mjlab/main/source/research.html) page for publications and projects, or share your own in [Show and Tell](https://github.com/mujocolab/mjlab/discussions/categories/show-and-tell).

If you use mjlab in your research, please consider citing:

```bibtex
@misc{zakka2026mjlablightweightframeworkgpuaccelerated,
  title={mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning},
  author={Kevin Zakka and Qiayuan Liao and Brent Yi and Louis Le Lay and Koushil Sreenath and Pieter Abbeel},
  year={2026},
  eprint={2601.22074},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.22074},
}
```

## License

mjlab is licensed under the [Apache License, Version 2.0](LICENSE).

### Third-Party Code

Some portions of mjlab are forked from external projects:

- **`src/mjlab/utils/lab_api/`** — Utilities forked from [NVIDIA Isaac
  Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause license, see file
  headers)

Forked components retain their original licenses. See file headers for details.

## Acknowledgments

mjlab wouldn't exist without the excellent work of the Isaac Lab team, whose API
design and abstractions mjlab builds upon.

Thanks to the MuJoCo Warp team — especially Erik Frey and Taylor Howell — for
answering our questions, giving helpful feedback, and implementing features
based on our requests countless times.
