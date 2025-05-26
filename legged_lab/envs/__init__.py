# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from legged_lab.envs.g1.g1_config import (
    G1FlatAgentCfg,
    G1FlatEnvCfg,
    G1RoughAgentCfg,
    G1RoughEnvCfg,
)
from legged_lab.envs.gr2.gr2_config import (
    GR2FlatAgentCfg,
    GR2FlatEnvCfg,
    GR2RoughAgentCfg,
    GR2RoughEnvCfg,
)
from legged_lab.envs.h1.h1_config import (
    H1FlatAgentCfg,
    H1FlatEnvCfg,
    H1RoughAgentCfg,
    H1RoughEnvCfg,
)
from legged_lab.envs.atom01.atom01_config import (
    ATOM01FlatAgentCfg,
    ATOM01FlatEnvCfg,
    ATOM01RoughAgentCfg,
    ATOM01RoughEnvCfg,
)
from legged_lab.utils.task_registry import task_registry

task_registry.register("h1_flat", BaseEnv, H1FlatEnvCfg(), H1FlatAgentCfg())
task_registry.register("h1_rough", BaseEnv, H1RoughEnvCfg(), H1RoughAgentCfg())
task_registry.register("g1_flat", BaseEnv, G1FlatEnvCfg(), G1FlatAgentCfg())
task_registry.register("g1_rough", BaseEnv, G1RoughEnvCfg(), G1RoughAgentCfg())
task_registry.register("gr2_flat", BaseEnv, GR2FlatEnvCfg(), GR2FlatAgentCfg())
task_registry.register("gr2_rough", BaseEnv, GR2RoughEnvCfg(), GR2RoughAgentCfg())
task_registry.register("atom01_rough", BaseEnv, ATOM01RoughEnvCfg(), ATOM01RoughAgentCfg())
task_registry.register("atom01_flat", BaseEnv, ATOM01FlatEnvCfg(), ATOM01FlatAgentCfg())
