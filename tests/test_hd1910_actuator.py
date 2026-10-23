"""HD1910 fit, robot-family selection, and Feetech controller lifecycle regressions."""

import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401 (load the task plugin before bam.mjlab)
from bam.feetech.actuator import STS3215Actuator
from bam.mjlab import BamActuator
from bam.model import load_model
from mjlab.entity import EntityCfg
from mjlab_microduck.actuator import FrictionDRBamActuator
from mjlab_microduck.robot import microduck_constants as robot

ROOT = Path(__file__).resolve().parents[1]
ROBOTS = [
    (name, cfg) for name, cfg in vars(robot).items()
    if name.endswith("_ROBOT_CFG") and isinstance(cfg, EntityCfg)
]


@pytest.mark.parametrize("name,cfg", ROBOTS, ids=[name for name, _ in ROBOTS])
def test_entire_robot_family_uses_1910_and_only_servo_joints(name, cfg):
    actuator, = cfg.articulation.actuators
    assert actuator.json_path == str(robot.HD1910_PARAMS_PATH)
    assert actuator.kp_fw == 5.0
    assert actuator.vin_range == (7.0, 7.6) and actuator.vin_min == 6.8
    assert (actuator.delay_min_lag, actuator.delay_max_lag) == (3, 6)
    model = cfg.spec_fn().compile()
    selected = [
        model.joint(j).name for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        and any(re.fullmatch(pattern, model.joint(j).name)
                for pattern in actuator.target_names_expr)
    ]
    assert len(selected) == 14
    assert not any(j.startswith("passive_") for j in selected)


def test_1910_json_loads_the_fitted_feetech_motor():
    params = json.loads(robot.HD1910_PARAMS_PATH.read_text())
    model = load_model(str(robot.HD1910_PARAMS_PATH))
    assert params["actuator"] == "sts3215" and params["model"] == "m6"
    assert isinstance(model.actuator, STS3215Actuator)
    assert model.kt.value == pytest.approx(0.6237611235393989)
    assert model.R.value == pytest.approx(4.910191564179625)
    assert model.actuator.get_extra_inertia() == pytest.approx(0.0017819049974475338)
    assert model.max_velocity.value == 100.0


def test_walk_models_match_measured_assembled_mass():
    for spec_fn in (robot.get_walk_spec, robot.get_walk_backlash_spec):
        model = spec_fn().compile()
        assert float(model.body_mass.sum()) == pytest.approx(
            robot.MICRODUCK_REAL_MASS_KG, abs=1e-6
        )


def test_training_feetech_goal_initialization_and_partial_reset(monkeypatch):
    motor = load_model(str(robot.HD1910_PARAMS_PATH)).actuator

    def initialize(self, *args):
        self._bam_model = motor.model
        self._num_envs = 2
        self._target_names = ["left_knee", "right_knee"]
        self.kp_scale = torch.ones(2, 1)

    monkeypatch.setattr(BamActuator, "initialize", initialize)
    monkeypatch.setattr(BamActuator, "reset", lambda self, env_ids: None)
    monkeypatch.setattr(
        BamActuator, "compute",
        lambda self, cmd: self._bam_model.actuator.q_target_smooth.clone(),
    )
    actuator = object.__new__(FrictionDRBamActuator)
    actuator.initialize(None, None, None, "cpu")
    pos = torch.tensor([[0.2, -0.3], [0.4, -0.5]])
    assert torch.equal(actuator.compute(SimpleNamespace(pos=pos)), pos)
    motor.q_target_smooth.fill_(1.0)
    actuator.reset(torch.tensor([0]))
    result = actuator.compute(SimpleNamespace(pos=pos * 2))
    assert torch.equal(result[0], pos[0] * 2)
    assert torch.equal(result[1], torch.ones(2))
    actuator.reset()
    assert torch.equal(actuator.compute(SimpleNamespace(pos=pos)), pos)


@pytest.fixture
def ip():
    spec = importlib.util.spec_from_file_location("infer_hd1910", ROOT / "scripts/infer_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_feetech_control_and_reset(ip):
    fit = ip.load_bam_model(ip.BAM_KP_FW, 7.4, None)
    model, data, ctrl, _ = ip.load_mujoco_with_bam(
        str(ROOT / ip.MICRODUCK_XML), fit, 0.005, 0.1, ip.BAM_VIN_MIN,
    )
    data.qpos[ctrl.qpos_indexes] = ip.DEFAULT_POSE
    ctrl.reset(data.qpos)
    assert np.allclose(fit.actuator.q_target_smooth, ip.DEFAULT_POSE)
    # Exercise the actual Feetech PWM law, including its fitted error-gain ratio.
    target = ip.DEFAULT_POSE + 0.01
    volts = fit.actuator.compute_control(target, ip.DEFAULT_POSE, np.zeros(14), 0.005)
    expected = 7.4 * 0.01 * 5.0 * 0.166 * fit.error_gain_ratio.value
    assert np.allclose(volts, expected, atol=1e-7)
    fit.actuator.q_target_smooth[:] = 2.0
    data.time = 3.0
    ctrl.last_ts = 3.0
    mujoco.mj_resetData(model, data)
    data.qpos[ctrl.qpos_indexes] = -ip.DEFAULT_POSE
    ctrl.reset(data.qpos)
    assert np.allclose(fit.actuator.q_target_smooth, -ip.DEFAULT_POSE)
    assert ctrl.last_ts == 0.0


def test_cpu_rejects_unimplemented_feetech_current_limit(ip):
    with pytest.raises(ValueError, match="does not implement"):
        ip.load_bam_model(ip.BAM_KP_FW, 7.4, 1.75)
