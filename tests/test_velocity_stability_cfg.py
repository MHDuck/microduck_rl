"""Regression tests for the hardware-driven stable walking recipe."""

import math

from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.robot import microduck_constants as robot
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    HEAD_ZERO_COMMAND_PROB,
    MicroduckRlCfg,
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)


def test_slow_forward_bucket_and_command_curriculum_are_enabled():
    cfg = make_microduck_velocity_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.rel_slow_forward_envs == pytest.approx(0.40)
    assert cmd.slow_forward_range == (0.10, 0.30)
    assert cmd.ranges.lin_vel_x == (-0.18, 0.18)
    assert cmd.ranges.lin_vel_y == (-0.04, 0.04)
    assert cmd.ranges.ang_vel_z == (-0.20, 0.20)

    stages = cfg.curriculum["command_velocity"].params["velocity_stages"]
    assert stages[0]["step"] == 0
    assert stages[-1] == {
        "step": 3000 * NUM_STEPS_PER_ENV,
        "lin_vel_x_range": 0.40,
        "lin_vel_y_range": 0.20,
        "ang_vel_range": 0.80,
    }


def test_reward_stack_prices_deadzone_thrashing_and_asymmetry():
    cfg = make_microduck_velocity_env_cfg()
    rewards = cfg.rewards
    assert rewards["track_linear_velocity"].weight == pytest.approx(4.0)
    assert rewards["track_linear_velocity"].params["std"] == pytest.approx(0.18)
    assert rewards["air_time"].weight == pytest.approx(1.35)
    assert rewards["air_time"].params["threshold_min"] == pytest.approx(0.160)
    assert rewards["air_time"].params["threshold_max"] == pytest.approx(0.340)
    assert rewards["body_ang_vel"].weight == pytest.approx(-0.15)
    assert rewards["gait_symmetry"].weight < 0.0
    assert rewards["leg_action_acceleration_l2"].weight == 0.0
    assert rewards["leg_joint_velocity_l2"].weight == 0.0
    assert rewards["heading_hold"].params["command_name"] == "twist"
    assert rewards["heading_hold"].weight == pytest.approx(2.0)
    assert rewards["heading_hold"].params["std"] == pytest.approx(0.14)
    assert rewards["heading_error_l1"].weight == pytest.approx(1.5)
    assert rewards["action_over_limit"].weight == pytest.approx(-1.0)
    assert rewards["action_over_limit"].params["overshoot"] == pytest.approx(0.10)
    assert rewards["leg_joint_limit_proximity"].weight == pytest.approx(-2.0)
    assert rewards["leg_joint_limit_proximity"].params["margin"] == pytest.approx(0.18)
    assert rewards["foot_clearance"].params["target_height"] == pytest.approx(0.018)
    assert rewards["foot_swing_height"].params["target_height"] == pytest.approx(0.018)

    stride = rewards["touchdown_stride"]
    assert stride.weight == 0.0
    assert stride.params["base_target_m"] == pytest.approx(0.035)
    assert stride.params["speed_target_s"] == pytest.approx(0.13)
    assert stride.params["max_target_m"] == pytest.approx(0.070)
    assert stride.params["min_air_time"] == pytest.approx(0.15)

    swing = rewards["swing_foot_lead"]
    assert swing.weight == 0.0
    assert swing.params["base_target_m"] == pytest.approx(0.035)
    assert swing.params["max_target_m"] == pytest.approx(0.070)
    assert rewards["swing_foot_progress"].weight == 0.0
    assert rewards["short_swing"].weight == 0.0

    accel = cfg.curriculum["leg_action_acceleration_weight"].params["weight_stages"]
    assert accel[0]["weight"] == 0.0
    assert accel[-1]["weight"] == pytest.approx(-0.12)
    velocity = cfg.curriculum["leg_joint_velocity_weight"].params["weight_stages"]
    assert velocity[0]["weight"] == 0.0
    assert velocity[-1]["weight"] == pytest.approx(-0.010)
    action_rate = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert action_rate[-1]["weight"] == pytest.approx(-1.10)

    stride_weights = cfg.curriculum["touchdown_stride_weight"].params["weight_stages"]
    assert stride_weights[0] == {"step": 0, "weight": 0.0}
    assert stride_weights[-1] == {
        "step": 1500 * NUM_STEPS_PER_ENV,
        "weight": 6.0,
    }
    swing_weights = cfg.curriculum["swing_foot_lead_weight"].params["weight_stages"]
    assert swing_weights[0] == {"step": 0, "weight": 0.0}
    assert swing_weights[-1] == {
        "step": 1500 * NUM_STEPS_PER_ENV,
        "weight": 2.0,
    }
    progress_weights = cfg.curriculum["swing_foot_progress_weight"].params["weight_stages"]
    assert progress_weights[-1] == {
        "step": 1000 * NUM_STEPS_PER_ENV,
        "weight": 4.0,
    }
    short_weights = cfg.curriculum["short_swing_weight"].params["weight_stages"]
    assert short_weights[-1] == {
        "step": 2500 * NUM_STEPS_PER_ENV,
        "weight": -3.0,
    }


def test_touchdown_stride_requires_a_real_single_touchdown_and_progress():
    command = torch.tensor([[0.15, 0.0, 0.0]])
    contact_time = torch.tensor([[0.02, 0.50]])
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            current_contact_time=contact_time,
            last_air_time=torch.tensor([[0.20, 0.0]]),
            current_air_time=torch.tensor([[0.20, 0.0]]),
        )
    )
    asset = SimpleNamespace(
        data=SimpleNamespace(
            site_pos_w=torch.tensor([[[0.02875, 0.0, 0.0], [-0.02875, 0.0, 0.0]]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_link_lin_vel_b=torch.tensor([[0.10, 0.0, 0.0]]),
        )
    )
    env = SimpleNamespace(
        scene={"feet": sensor, "robot": asset},
        command_manager=SimpleNamespace(get_command=lambda _name: command),
        episode_length_buf=torch.tensor([10]),
        step_dt=0.02,
        num_envs=1,
    )
    asset_cfg = SimpleNamespace(name="robot", site_ids=[0, 1])

    reward = microduck_mdp.touchdown_stride_reward(
        env,
        sensor_name="feet",
        asset_cfg=asset_cfg,
    )
    assert reward.item() == pytest.approx(1.0)

    # At vx=0.15 the configured target is 3.5 cm + 0.15 s * 0.15 m/s
    # = 5.75 cm. A left swing foot that far ahead receives the full dense score.
    swing_reward = microduck_mdp.swing_foot_lead_reward(
        env,
        sensor_name="feet",
        asset_cfg=asset_cfg,
    )
    assert swing_reward.item() == pytest.approx(1.0)

    # Progress shaping starts at zero on liftoff, then immediately pays forward
    # displacement even before touchdown.
    assert microduck_mdp.swing_foot_progress_reward(
        env, sensor_name="feet", asset_cfg=asset_cfg
    ).item() == pytest.approx(0.0)
    asset.data.site_pos_w[:, 0, 0] += 0.01
    assert microduck_mdp.swing_foot_progress_reward(
        env, sensor_name="feet", asset_cfg=asset_cfg
    ).item() > 0.0

    # A split stance without commanded-direction progress cannot farm the
    # event reward by tapping a foot in place.
    asset.data.root_link_lin_vel_b.zero_()
    assert microduck_mdp.touchdown_stride_reward(
        env, sensor_name="feet", asset_cfg=asset_cfg
    ).item() == pytest.approx(0.0)


def test_heading_l1_keeps_corrective_gradient_only_for_straight_motion():
    command = torch.tensor([[0.20, 0.0, 0.0]])
    asset = SimpleNamespace(
        data=SimpleNamespace(root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    )
    env = SimpleNamespace(
        scene={"robot": asset},
        command_manager=SimpleNamespace(get_command=lambda _name: command),
        episode_length_buf=torch.tensor([0]),
        num_envs=1,
    )
    asset_cfg = SimpleNamespace(name="robot")

    # Capture the reset heading, then veer by 0.3 rad.
    assert microduck_mdp.heading_error_l1_penalty(
        env, asset_cfg=asset_cfg
    ).item() == pytest.approx(0.0)
    env.episode_length_buf[:] = 10
    half = 0.3 / 2.0
    asset.data.root_link_quat_w[:] = torch.tensor(
        [[math.cos(half), 0.0, 0.0, math.sin(half)]]
    )
    assert microduck_mdp.heading_error_l1_penalty(
        env, asset_cfg=asset_cfg
    ).item() == pytest.approx(-0.3, abs=1e-5)

    # A deliberate yaw command gates the straight-line penalty away.
    command[:, 2] = 0.5
    assert abs(microduck_mdp.heading_error_l1_penalty(
        env, asset_cfg=asset_cfg
    ).item()) < 1e-6

    # Simultaneous touchdown (e.g. reset/landing) is excluded as well.
    asset.data.root_link_lin_vel_b[:, 0] = 0.10
    sensor.data.current_contact_time[:] = 0.02
    sensor.data.last_air_time[:] = 0.20
    assert microduck_mdp.touchdown_stride_reward(
        env, sensor_name="feet", asset_cfg=asset_cfg
    ).item() == pytest.approx(0.0)


def test_stride_tuning_can_be_disabled_for_derived_tasks():
    cfg = make_microduck_velocity_env_cfg(stride_tuning=False)
    assert "touchdown_stride" not in cfg.rewards
    assert "swing_foot_lead" not in cfg.rewards
    assert "swing_foot_progress" not in cfg.rewards
    assert "short_swing" not in cfg.rewards
    assert "touchdown_stride_weight" not in cfg.curriculum
    assert "swing_foot_lead_weight" not in cfg.curriculum
    assert "swing_foot_progress_weight" not in cfg.curriculum
    assert "short_swing_weight" not in cfg.curriculum
    assert cfg.commands["twist"].rel_slow_forward_envs == pytest.approx(0.40)
    assert cfg.commands["twist"].slow_forward_range == (0.06, 0.20)
    assert cfg.rewards["air_time"].params["threshold_min"] == pytest.approx(0.125)
    assert cfg.rewards["air_time"].params["threshold_max"] == pytest.approx(0.300)
    assert cfg.rewards["foot_clearance"].params["target_height"] == pytest.approx(0.014)
    assert cfg.rewards["foot_swing_height"].params["target_height"] == pytest.approx(0.014)
    assert cfg.curriculum["action_rate_weight"].params["weight_stages"][-1]["weight"] == pytest.approx(-1.2)
    assert cfg.curriculum["leg_action_acceleration_weight"].params["weight_stages"][-1]["weight"] == pytest.approx(-0.10)

def test_pushes_start_at_zero_and_are_disabled_in_play():
    train = make_microduck_velocity_env_cfg()
    play = make_microduck_velocity_env_cfg(play=True)
    assert "push_robot" in train.events
    assert "push_robot" not in play.events
    stages = train.curriculum["push_magnitude"].params["push_stages"]
    assert stages[0]["velocity_range"] == {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
    }
    assert stages[-1]["velocity_range"] == {
        "x": (-0.15, 0.15),
        "y": (-0.15, 0.15),
    }
    assert "push_magnitude" not in play.curriculum

    # Construction order must not matter: nested configs may be shared by the
    # upstream factory unless this task clones them before mutation.
    play_first = make_microduck_velocity_env_cfg(play=True)
    train_after = make_microduck_velocity_env_cfg()
    assert "push_robot" not in play_first.events
    assert "push_magnitude" not in play_first.curriculum
    assert "push_robot" in train_after.events


def test_hardware_envelope_and_head_com_selection_match_measurements():
    actuator, = robot.MICRODUCK_WALK_ROBOT_CFG.articulation.actuators
    assert actuator.kp_fw == pytest.approx(5.0)
    assert actuator.vin_range == (7.0, 7.6)
    assert actuator.vin_min == pytest.approx(6.8)
    assert robot.MICRODUCK_REAL_MASS_KG == pytest.approx(0.821)

    cfg = make_microduck_velocity_env_cfg()
    head_bodies = cfg.events["randomize_head_com"].params["asset_cfg"].body_names
    assert "bearing_roll" not in head_bodies
    assert cfg.observations["actor"].terms["joint_vel"].noise.n_max == pytest.approx(0.08)
    assert cfg.events["foot_friction"].params["ranges"] == (0.6, 1.8)
    assert robot.HOME_FRAME.joint_pos[r".*neck_pitch.*"] == pytest.approx(0.3491)
    assert robot.HOME_FRAME.joint_pos[r".*head_pitch.*"] == pytest.approx(0.3491)
    assert cfg.rewards["head_pose_bias"].params["component_indices"] == (0, 1)
    assert cfg.commands["head_pose"].zero_command_prob == pytest.approx(
        HEAD_ZERO_COMMAND_PROB
    )
    assert HEAD_ZERO_COMMAND_PROB == pytest.approx(0.60)


def test_stable_recipe_uses_a_fresh_checkpoint_namespace():
    assert MicroduckRlCfg.experiment_name == "velocity_hd1910_stride_v3"
    assert MicroduckRlCfg.run_name == "straight_head_finetune_v5"
    assert MicroduckRlCfg.max_iterations == 6_000
    assert MicroduckRlCfg.algorithm.symmetry_cfg is not None
    assert MicroduckRlCfg.algorithm.symmetry_cfg["use_mirror_loss"] is True
