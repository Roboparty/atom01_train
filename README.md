# roboparty_train

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-main-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL_RL](https://img.shields.io/badge/RSL_RL-3.3.0-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

[English](README.md) | [中文](README_CN.md)

## Overview

`roboparty_train` is the Roboparty training workspace for RPO locomotion policies. It keeps Roboparty task definitions, motion-retargeting tools, MuJoCo Sim2Sim scripts, and RSL-RL training entry points outside the upstream Isaac Lab tree.

The repository is organized as a thin workspace with two Git submodules:

- `robolab`: Roboparty Isaac Lab extension, environments, scripts, and robot assets.
- `rsl_rl`: Roboparty-compatible RSL-RL dependency snapshot.

**Maintainer**: RoboParty
**Support**: GitHub Issues

## Features

- RPO training environments built on Isaac Lab.
- RSL-RL training, evaluation, and policy export entry points.
- AMP, BeyondMimic, and Parkour workflows.
- MuJoCo Sim2Sim scripts for policy transfer checks.
- Motion retargeting utilities for datasets prepared with [GMR](https://github.com/Roboparty/GMR).

## Requirements

- Python 3.11.
- Isaac Sim 5.1.0 and Isaac Lab main.
- Ubuntu 22.04 x64 or Windows x64.

Install Isaac Lab by following the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). Run the commands below in the Python environment used by Isaac Lab.

## Clone

Clone this repository outside the Isaac Lab source tree. Use `--recursive` so that `robolab` and `rsl_rl` are populated immediately:

```bash
git clone --recursive https://github.com/Roboparty/roboparty_train.git
cd roboparty_train
```

If you already cloned the repository and `robolab` or `rsl_rl` is empty, initialize the submodules manually:

```bash
cd roboparty_train
git submodule update --init --recursive
```

## Install

```bash
pip install -e ./robolab
pip install -e ./rsl_rl
```

Verify that the extension is installed by listing the available environments:

```bash
python robolab/scripts/tools/list_envs.py
```

## Usage

### Train

```bash
python robolab/scripts/rsl_rl/train.py --task=<ENV_NAME> --headless --logger=tensorboard --num_envs=8192
```

### Play

```bash
python robolab/scripts/rsl_rl/play.py --task=<ENV_NAME> --num_envs=1
```

### Play AMP

```bash
python robolab/scripts/rsl_rl/play_amp.py --task=RPO-AMP-Play --num_envs=1
```

### Play BeyondMimic

```bash
python robolab/scripts/rsl_rl/play_bm.py --task=RPO-BeyondMimic --num_envs=1
```

### Play Parkour

```bash
python robolab/scripts/rsl_rl/play_parkour.py --task=RPO-Parkour-Play --num_envs=1
```

To export an ONNX model, set `num_envs=1` and add `--exportonnx`:

```bash
python robolab/scripts/rsl_rl/play_parkour.py --task=RPO-Parkour-Play --num_envs=1 --exportonnx
```

### Sim2Sim

```bash
python robolab/scripts/mujoco/sim2sim_rpo.py --load_model "{exported/policy.pt model full path here}"
```

### Prepare Motion Data

To prepare datasets for AMP and BeyondMimic, see [GMR](https://github.com/Roboparty/GMR).

The joint order in datasets produced by GMR follows the robot URDF/XML order, which differs from the order used by Isaac Lab. Prepare a joint mapping file such as `robolab/scripts/tools/retarget/config/rpo.yaml`, then reorder the joint sequence before training:

```bash
python robolab/scripts/tools/retarget/dataset_retarget.py
```

## Troubleshooting

- Empty `robolab` or `rsl_rl` directory: run `git submodule update --init --recursive`.
- Missing Isaac Lab imports: activate the Python environment used by Isaac Lab before installing or running scripts.
- Missing RPO task name: run `python robolab/scripts/tools/list_envs.py` and copy the exact task id.

## References and Thanks

This project builds on the following open-source projects:

- [IsaacLab](https://github.com/isaac-sim/IsaacLab)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [legged_lab](https://github.com/zitongbai/legged_lab)
- [robot_lab](https://github.com/fan-ziqi/robot_lab)
- [InstinctLab](https://github.com/project-instinct/InstinctLab)
