"""Regression coverage for the gentle HD1910 sit/stand recipe."""

import pytest

from mjlab_microduck.tasks.microduck_sitstand_env_cfg import (
    MAX_DESCENT_SPEED,
    MAX_RISE_SPEED,
    MicroduckSitStandRlCfg,
    POSTURE_RAMP_S,
    make_microduck_sitstand_env_cfg,
)
from mjlab_microduck.tasks.microduck_standup_env_cfg import make_microduck_standup_env_cfg


def test_transition_is_slow_and_head_impact_is_measured():
    cfg = make_microduck_sitstand_env_cfg()
    assert POSTURE_RAMP_S == pytest.approx(3.0)
    assert MAX_DESCENT_SPEED == pytest.approx(0.035)
    assert MAX_RISE_SPEED == pytest.approx(0.07)
    assert "head_ground_contact" in {sensor.name for sensor in cfg.scene.sensors}
    impact = cfg.rewards["head_impact"]
    assert impact.weight == pytest.approx(-0.25)
    assert impact.params == {"sensor_name": "head_ground_contact", "threshold": 1.0}


def test_head_has_a_strong_endpoint_and_a_delayed_slew_cost():
    cfg = make_microduck_sitstand_env_cfg()
    assert cfg.rewards["head_pose_tracking"].weight == pytest.approx(1.5)
    assert cfg.rewards["posture_composite"].params["head_std"] == pytest.approx(0.30)
    assert cfg.rewards["neck_action_rate_l2"].weight == 0.0
    stages = cfg.curriculum["neck_action_rate_weight"].params["weight_stages"]
    assert stages[0]["weight"] == 0.0
    assert stages[-1]["weight"] == pytest.approx(-0.35)


def test_hardware_noise_and_gentleness_weights_are_locked():
    cfg = make_microduck_sitstand_env_cfg()
    noise = cfg.observations["actor"].terms["joint_vel"].noise
    assert noise.n_min == pytest.approx(-0.08)
    assert noise.n_max == pytest.approx(0.08)
    assert cfg.rewards["gentle_motion"].weight == pytest.approx(0.10)
    assert cfg.events["foot_friction"].params["ranges"] == (0.6, 1.8)
    descent = cfg.curriculum["descent_speed_weight"].params["weight_stages"]
    assert descent[-1]["weight"] == pytest.approx(30.0)


def test_recovery_policy_covers_the_same_floor_to_carpet_envelope():
    cfg = make_microduck_standup_env_cfg()
    assert cfg.events["foot_friction"].params["ranges"] == (0.6, 1.8)


def test_gentle_recipe_uses_a_fresh_checkpoint_namespace():
    assert MicroduckSitStandRlCfg.experiment_name == "microduck_sitstand_gentle_v2"
    assert MicroduckSitStandRlCfg.run_name == "microduck_sitstand_gentle_v2"
