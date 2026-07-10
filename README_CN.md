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

## 概述

`roboparty_train` 是 Roboparty 面向 RPO 运动控制策略的训练工作区。它把 Roboparty 的任务定义、动作重定向工具、MuJoCo Sim2Sim 脚本和 RSL-RL 训练入口放在上游 Isaac Lab 源码树之外，便于独立维护和同步。

本仓库是一个轻量工作区，包含两个 Git submodule：

- `robolab`：Roboparty 的 Isaac Lab 扩展、环境、脚本和机器人资产。
- `rsl_rl`：与 Roboparty 训练流程兼容的 RSL-RL 依赖快照。

**维护者**: RoboParty
**支持渠道**: GitHub Issues

## 主要特性

- 基于 Isaac Lab 的 RPO 训练环境。
- RSL-RL 训练、评估和策略导出入口。
- AMP、BeyondMimic 和 Parkour 工作流。
- 用于策略迁移检查的 MuJoCo Sim2Sim 脚本。
- 面向 [GMR](https://github.com/Roboparty/GMR) 数据集的动作重定向工具。

## 环境要求

- Python 3.11。
- Isaac Sim 5.1.0 和 Isaac Lab main。
- Ubuntu 22.04 x64 或 Windows x64。

请按照 Isaac Lab 的[官方安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)安装 Isaac Lab。下面的命令需要在 Isaac Lab 使用的 Python 环境中执行。

## 克隆仓库

请把本仓库克隆到 Isaac Lab 源码树之外。推荐使用 `--recursive`，这样 `robolab` 和 `rsl_rl` 会被一起拉取：

```bash
git clone --recursive https://github.com/Roboparty/roboparty_train.git
cd roboparty_train
```

如果已经普通克隆过仓库，并且看到 `robolab` 或 `rsl_rl` 是空目录，请手动初始化 submodule：

```bash
cd roboparty_train
git submodule update --init --recursive
```

## 安装

```bash
pip install -e ./robolab
pip install -e ./rsl_rl
```

通过列出环境确认扩展安装成功：

```bash
python robolab/scripts/tools/list_envs.py
```

## 用法

### 训练

```bash
python robolab/scripts/rsl_rl/train.py --task=<ENV_NAME> --headless --logger=tensorboard --num_envs=8192
```

### 测试

```bash
python robolab/scripts/rsl_rl/play.py --task=<ENV_NAME> --num_envs=1
```

### 测试 AMP

```bash
python robolab/scripts/rsl_rl/play_amp.py --task=RPO-AMP-Play --num_envs=1
```

### 测试 BeyondMimic

```bash
python robolab/scripts/rsl_rl/play_bm.py --task=RPO-BeyondMimic --num_envs=1
```

### 测试 Parkour

```bash
python robolab/scripts/rsl_rl/play_parkour.py --task=RPO-Parkour-Play --num_envs=1
```

导出 ONNX 模型时，请设置 `num_envs=1` 并添加 `--exportonnx`：

```bash
python robolab/scripts/rsl_rl/play_parkour.py --task=RPO-Parkour-Play --num_envs=1 --exportonnx
```

### Sim2Sim

```bash
python robolab/scripts/mujoco/sim2sim_rpo.py --load_model "{exported/policy.pt model full path here}"
```

### 准备动作数据

AMP 和 BeyondMimic 的数据集准备流程请参考 [GMR](https://github.com/Roboparty/GMR)。

GMR 生成的数据集关节顺序与机器人 URDF/XML 中的顺序一致，但这与 Isaac Lab 使用的顺序不同。训练前需要准备类似 `robolab/scripts/tools/retarget/config/rpo.yaml` 的关节映射文件，然后重新排列关节序列：

```bash
python robolab/scripts/tools/retarget/dataset_retarget.py
```

## 常见问题

- `robolab` 或 `rsl_rl` 是空目录：执行 `git submodule update --init --recursive`。
- 找不到 Isaac Lab 相关 import：请先激活 Isaac Lab 使用的 Python 环境，再安装或运行脚本。
- 找不到 RPO task 名称：执行 `python robolab/scripts/tools/list_envs.py`，复制实际输出的 task id。

## 参考和致谢

本项目基于以下开源项目构建：

- [IsaacLab](https://github.com/isaac-sim/IsaacLab)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [legged_lab](https://github.com/zitongbai/legged_lab)
- [robot_lab](https://github.com/fan-ziqi/robot_lab)
- [InstinctLab](https://github.com/project-instinct/InstinctLab)
