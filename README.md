# Legged Lab: Direct IsaacLab Workflow for Legged Robots

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL_RL](https://img.shields.io/badge/RSL_RL-2.3.1-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

## Overview

This repository provides a direct workflow for training a legged robot using IsaacLab. It provides high transparency and low refactoring difficulty of the direct environment, and uses isaaclab components to simplify the workflow.

It has all the necessary conditions for sim-to-real and has been tested on real unitree g1 and h1 robots, [video available](https://www.bilibili.com/video/BV1tNRgYQEnr/).
Deploy Code: https://github.com/Hellod035/LeggedLabDeploy

**Maintainer**: Wandong Sun
**Contact**: 24b908020@stu.hit.edu.cn

**Key Features:**

- `Easy to Reorganize` Provides a direct workflow, allowing for fine-grained definition of environment logic.
- `Isolation` Work outside the core Isaac Lab repository, ensuring that the development efforts remain self-contained.
- `Long-term support` This repository will be updated with the updates of isaac sim and isaac lab, and will be supported for a long time.



## Installation

LeggedLab is built against the latest version of Isaacsim/IsaacLab. It is recommended to follow the latest updates of legged lab.

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
# Option 1: HTTPS
git clone https://github.com/Hellod035/LeggedLab

# Option 2: SSH
git clone git@github.com:Hellod035/LeggedLab.git
```

- Using a python interpreter that has Isaac Lab installed, install the library

```bash
cd LeggedLab
pip install -e .
```

- Verify that the extension is correctly installed by running the following command:

```bash
python legged_lab/scripts/train.py --task=g1_flat --headless --logger=tensorboard --num_envs=64
```


## Use Your Own Robot

Assets must be converted into USD format to be compatible with Legged Lab/IsaacLab. [Convert Tutorial](https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html).


## Multi-GPU and Multi-Node Training

Legged Lab supports multi-GPU and multi-node reinforcement learning using rsl_rl, the usage is exactly the same as IsaacLab. [Detailed information](https://isaac-sim.github.io/IsaacLab/main/source/features/multi_gpu.html)

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "${workspaceFolder}/legged_lab",
        "<path-to-IsaacLab>/source/isaaclab_tasks",
        "<path-to-IsaacLab>/source/isaaclab_mimic",
        "<path-to-IsaacLab>/source/extensions",
        "<path-to-IsaacLab>/source/isaaclab_assets",
        "<path-to-IsaacLab>/source/isaaclab_rl",
        "<path-to-IsaacLab>/source/isaaclab",
    ]
}
```

## References and Thanks
This project repository builds upon the shoulders of giants.
* [IsaacLab](https://github.com/isaac-sim/IsaacLab)   The various reusable practical components in IsaacLab greatly simplify the complexity of LeggedLab.
* [legged_gym](https://github.com/leggedrobotics/legged_gym)   We borrowed the code organization and environment definition logic of legged_gym and simplified it as much as possible.
* [Protomotions](https://github.com/NVlabs/ProtoMotions)   The motivation for building this repository comes from protomotions. For the first time, we realized that we could create our own environment using only IsaacLab components without inheriting 'DirectRLEnv' or 'ManagerBasedRLEnv'.

## Citation

If you use Legged Lab in your research, you can cite it as follows:

```bibtex
@software{LeggedLab,
  author = {Wandong, Sun},
  license = {BSD-3-Clause},
  title = {Legged Lab: Direct IsaacLab Workflow for Legged Robots},
  url = {https://github.com/Hellod035/LeggedLab},
  version = {1.0.0},
  year = {2025}
}
```
