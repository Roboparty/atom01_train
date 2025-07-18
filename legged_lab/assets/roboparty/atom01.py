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


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

ATOM01_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/roboparty/atom/atom01.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),
        joint_pos={
            "left_thigh_pitch_joint": -0.24,
            "left_knee_joint": 0.48,
            "left_ankle_pitch_joint": -0.24,
            "left_arm_pitch_joint": 0.1,
            "left_arm_roll_joint": 0.07,
            "left_elbow_pitch_joint": 1.2,
            "right_thigh_pitch_joint": -0.24,
            "right_knee_joint": 0.48,
            "right_ankle_pitch_joint": -0.24,
            "right_arm_pitch_joint": 0.1,
            "right_arm_roll_joint": -0.07,
            "right_elbow_pitch_joint": 1.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_thigh_yaw_joint",
                ".*_thigh_roll_joint",
                ".*_thigh_pitch_joint",
                ".*_knee_joint",
                ".*torso.*",
            ],
            stiffness={
                ".*_thigh_yaw_joint": 150.0,
                ".*_thigh_roll_joint": 150.0,
                ".*_thigh_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                ".*torso.*": 200.0,
            },
            damping={
                ".*_thigh_yaw_joint": 5.0,
                ".*_thigh_roll_joint": 5.0,
                ".*_thigh_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                ".*torso.*": 5.0,
            },
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=50.0,
            damping=2.5,
            armature=0.01,
        ),
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_arm_pitch_joint",
                ".*_arm_roll_joint",
                ".*_arm_yaw_joint",
            ],
            stiffness=75.0,
            damping=3.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            stiffness={
                ".*_elbow_pitch_joint": 50.0,
                ".*_elbow_yaw_joint": 25.0,
            },
            damping={
                ".*_elbow_pitch_joint": 2.5,
                ".*_elbow_yaw_joint": 2.0,
            },
            armature=0.01,
        ),
    },
)
