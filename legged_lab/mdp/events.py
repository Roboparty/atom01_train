from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv

def randomize_base_body_com(
    env: BaseEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_range: dict[str, tuple[float, float]],
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # obtain the coms of the bodies
    coms = asset.root_physx_view.get_coms()

    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 1, 3), device=asset.device
    )

    coms_default = coms[env_ids[:, None], body_ids, :3].clone()
    coms[env_ids[:, None], body_ids, :3] = coms_default + rand_samples.cpu()

    # set the coms into the physics simulation
    asset.root_physx_view.set_coms(coms, env_ids)