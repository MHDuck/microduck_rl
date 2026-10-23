"""Microduck velocity (walking) environment.

The main locomotion task: velocity-command tracking + head-pose commands.
The reward/regularization recipe is locomotion-focused (lean tracking +
gait/feet terms, curriculum-ramped action-rate smoothing), with:

  - a dedicated 0.10--0.30 m/s forward bucket for the real robot's slow gait
  - staged command, push and smoothness curricula so balance is learned first
  - touchdown stride shaping, modest foot lift and explicit gait symmetry to
    suppress the high-cadence short-step gait seen on hardware
  - head_pose_tracking as a primary objective, plus an EMA-based head_pose_bias
    penalty that prices only the escapable DC head droop (see below)
  - body_pose tracking infra kept intact but DISABLED (weight 0) so the obs
    slot stays alive for envs that use it
"""

import math
from copy import deepcopy

NUM_STEPS_PER_ENV = 24

# Fraction of envs commanded to spin on the spot (lin=0, |ang| ∈ [0.4·max, max]).
TURN_IN_PLACE_FRACTION = 0.08
SLOW_FORWARD_FRACTION = 0.40

# Deployment sends an exact-zero head command during ordinary keyboard walking.
# A continuous uniform distribution almost never samples that point, which left
# the most common real command out-of-distribution and allowed a persistent
# upward head bias.  Keep non-zero head-pose control alive, but make the nominal
# walking case the majority of resamples.
HEAD_ZERO_COMMAND_PROB = 0.60

# Symmetry
ENABLE_SYMMETRY = True

# Domain randomization toggles
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True  # Randomize CoM of the head assembly bodies
ENABLE_KP_RANDOMIZATION = False # Was True
ENABLE_KD_RANDOMIZATION = False # Was True
ENABLE_MASS_INERTIA_RANDOMIZATION = True  # Can enable once walking is stable
ENABLE_JOINT_FRICTION_RANDOMIZATION = True  # Scales BAM's friction budget per-env via FrictionDRBamActuator.friction_scale
ENABLE_JOINT_DAMPING_RANDOMIZATION = False
ENABLE_ARMATURE_RANDOMIZATION = True  # Reflected rotor inertia (microban-style). DOES affect BAM (armature is set, not zeroed).
ENABLE_VELOCITY_PUSHES = True  # Velocity-based pushes for robustness training
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # Simulates mounting errors
ENABLE_ENCODER_BIAS = True  # Per-env joint encoder calibration offset (actor obs sees joint_pos + bias)
ENABLE_BASE_ORIENTATION_RANDOMIZATION = False  # Randomize initial tilt to force reactive behavior

# Head/body pose command tracking (replaces the old neck-offset disturbance scheme).
# Head pose: 4D deltas-from-HOME on neck/head joints; vel env tracks these as a
# primary objective. Body pose: 6D delta in [x, y, z, roll, pitch, yaw]; vel env
# samples small ranges + tiny reward weight so input neurons stay alive but
# tracking isn't the priority (standup env raises the weight).
HEAD_POSE_CMD_RESAMPLE_S = (2.0, 5.0)
BODY_POSE_CMD_RESAMPLE_S = (2.0, 5.0)

# Observation configuration
USE_PROJECTED_GRAVITY = True  # If True, use projected gravity instead of raw accelerometer

# Domain randomization ranges (adjust as needed)
# Conservative ranges proven to be stable - can increase gradually if needed
COM_RANDOMIZATION_RANGE = 0.003  # ±3mm initial, ramped to ±8mm via curriculum
# Head CoM randomization: applied per-episode to every body of the head assembly
# (neck → neck_pitch → yaw_roll_motion → head-roll body). Same non-accumulating
# mechanism as the trunk CoM randomization above. The head-roll body is named
# bottom_head_shell in the walk model and jaw_soft in the 2026-07 roller model,
# hence the alternation. ``bearing_roll`` is deliberately excluded: despite its
# name it is the right-hip-yaw link, and randomizing it here created a one-sided
# mass perturbation that matched the yaw/lateral drift seen on hardware.
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # ±3mm initial, ramped via curriculum
HEAD_BODY_NAMES = (
    "neck",
    "neck_pitch",
    "yaw_roll_motion",
    "(bottom_head_shell|jaw_soft)",
)
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)  # ±5% applied to BOTH mass and inertia together.
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # ±15%
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # ±10% (can increase to 0.8-1.2)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_DAMPING_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)  # ±10% reflected rotor inertia (microban: dr.joint_armature, same range)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)  # Apply pushes every 3-6 seconds
VELOCITY_PUSH_RANGE = (-0.15, 0.15)  # Final curriculum range; starts at zero.
# Large additive kicks from iteration zero produced a permanently nervous
# recovery gait. The curriculum below reaches this conservative final range only
# after clean low-speed locomotion has consolidated.
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0  # up-to-6° random-axis IMU mounting error. NOTE: zero-centered (random axis) — trains tolerance to misalignment *magnitude*, NOT a pitch bias. The real board's systematic ~5° pitch offset is corrected at the source in the runtime (imu-pitch-offset), not here.
ENCODER_BIAS_RANGE = (-0.015, 0.015)  # ±0.86° per-joint encoder offset (constant per env)
BASE_ORIENTATION_MAX_PITCH_DEG = 10.0  # ±10° forward/backward tilt at episode start
BASE_ORIENTATION_MAX_ROLL_DEG = 5.0  # ±5° side-to-side tilt at episode start

import mujoco as _mujoco
import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


# Microduck-specific rough terrain: much gentler than the default ROUGH_TERRAINS_CFG.
# The robot can only lift its feet ~1-2 cm, so steps are capped at 1.5 cm.
MICRODUCK_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    sub_terrains={
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.25),
        "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.0, 0.015),  # max 1.5 cm (vs 10 cm default)
            step_width=0.15,
            platform_width=2.0,
            border_width=1.0,
        ),
        # NOTE: BoxInvertedPyramidStairsTerrainCfg removed — it sets env_origin_z to the pit
        # bottom (negative), causing resets at root_z = 0.12 + env_origin_z ≈ −0.10 m which
        # places the robot below the pit floor and makes it fall through the ground.
        # Uneven cobblestone-like ground: random per-cell height offsets.
        # grid_width=0.12 on an 8m patch = 66×66 = 4 356 boxes/patch → ~261 K total → OOM.
        # 0.45 m gives 17×17 = 289 boxes/patch → ~17 K total (border = 0.35 m ✓).
        # Must not divide evenly into terrain size (8.0 m): 0.45 × 17 = 7.65 ✓
        "random_grid": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.30,
            grid_width=0.45,
            grid_height_range=(0.0, 0.010),  # max 1 cm
            platform_width=1.5,
        ),
        # Gentle slopes (heightfield pyramid, platform on TOP — robot spawns on
        # the flat platform and walks down/up/across the slope as commands
        # resample). slope_range is rise/run: 0.03→0.10 ≈ 1.7°→5.7° by
        # difficulty — small robot, small slopes. NOT inverted (see the
        # inverted-pyramid env_origin note above — same pit-spawn risk class).
        # vertical_scale=0.001 keeps quantization steps at 1 mm so a gentle
        # slope is smooth instead of a staircase of 5 mm ledges.
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.03, 0.10),
            platform_width=2.0,
            vertical_scale=0.001,
        ),
    },
    add_lights=False,
)


def _soften_terrain_contacts(spec: _mujoco.MjSpec) -> None:
    """Soften terrain box geom contacts to reduce edge-contact NaN instability.

    Box terrains place adjacent geoms at different heights. The hard edges where
    heights change cause contact normal instability when feet land on them, which
    can produce impulsive NaN forces in the MuJoCo solver.

    Doubling the solref time constant (0.02 → 0.04 s) makes contact springs
    2× softer — enough to damp the instability without noticeably changing the
    macro-level walking physics. Applied to all geoms in the "terrain" body,
    which contains every box generated by TerrainGenerator.
    """
    body = spec.body("terrain")
    count = 0
    for geom in body.geoms:
        geom.solref = [0.04, 1.0]   # 2× softer time constant (default: 0.02)
        geom.solimp = [0.85, 0.95, 0.001, 0.5, 2.0]  # slightly softer impedance
        count += 1
    print(f"[rough terrain] spec_fn: softened {count} terrain geoms (solref=0.04)")


def make_microduck_velocity_env_cfg(
    play: bool = False,
    rough: bool = False,
    stride_tuning: bool = True,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck velocity tracking environment configuration.

    ``stride_tuning=False`` preserves the pre-stride recipe for derived tasks
    (notably VelStand), so tuning the main walking task cannot silently alter an
    already validated recovery-policy training distribution.
    """

    std_standing = {
        # Lower body — tighter to keep the robot in home pose when standing
        r".*hip_yaw.*": 0.1,
        r".*hip_roll.*": 0.05,  # 0.1→0.06→0.05 — hold the 5°-inward stance (sole sits flat), stop leg splay
        r".*hip_pitch.*": 0.15,
        r".*knee.*": 0.15,
        r".*ankle.*": 0.1,
    }

    std_walking = {
        # Lower body
        r".*hip_yaw.*": 0.28 if stride_tuning else 0.3,
        r".*hip_roll.*": 0.05,  # 0.1→0.06→0.05 — hold the 5°-inward stance, stop the leg splay to vertical
        r".*hip_pitch.*": 0.42 if stride_tuning else 0.4,
        r".*knee.*": 0.42 if stride_tuning else 0.4,
        r".*ankle.*": 0.28 if stride_tuning else 0.25,
    }

    site_names = ["left_foot", "right_foot"]

    # Contact sensor for feet - LEFT, RIGHT order
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",  # LEFT foot first, RIGHT foot second
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # mjlab 1.3.0: foot_height obs + foot_clearance/foot_swing_height rewards are
    # now driven by a per-foot terrain-height ray sensor (was site_pos based).
    # Mirrors microban's foot_height_scan.
    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in site_names),
        pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=2),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )

    foot_frictions_geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )

    # Base configuration
    cfg = make_velocity_env_cfg()
    # mjlab's base factory retains shared references for several nested config
    # sections. Copy before the first mutation so constructing train/play or a
    # derived task in any order cannot leak events, rewards or curricula into
    # another instance (notably train's push_robot appearing in play).
    cfg.scene = deepcopy(cfg.scene)
    cfg.actions = deepcopy(cfg.actions)
    cfg.rewards = deepcopy(cfg.rewards)
    cfg.events = deepcopy(cfg.events)
    cfg.curriculum = deepcopy(cfg.curriculum)
    cfg.commands = deepcopy(cfg.commands)
    cfg.observations = deepcopy(cfg.observations)

    # Robot setup
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, foot_height_scan_cfg)
    cfg.viewer.body_name = "trunk_base"

    # Action configuration
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # === REWARDS ===
    # Pose reward configuration
    cfg.rewards["pose"].params["std_standing"] = std_standing  # tight when command=0
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking
    # Pose reward operates on LEG joints only. Head/neck are command-driven
    # (head_pose_tracking) — if they were in this reward too, it would pull
    # them to HOME while head_pose_tracking pulls them to the command, and the
    # policy converges to "ignore the command" because pose reward dominates
    # once head_pose_tracking's gradient dies at large commands.
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
    )
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].weight = 1.0

    # Body-specific reward configurations
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    # upright: deliberately strong (2.0 / std²=0.05, was 1.0 / std²=0.1).
    # 2026-07 pitch-vs-speed eval: the policy walks with a +2-4° steady forward
    # lean (p90 ~6-8°) and ~2/3 of push-induced falls at speed are FORWARD. At
    # weight 1.0 / std²=0.1 a 4° lean cost ~0.05/step — effectively free. At
    # 2.0 / std²=0.05 it costs ~0.19/step: enough gradient to hold the trunk
    # level in steady gait while transient lean (push recovery, accel) stays
    # affordable.
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    # Foot-specific configurations. In mjlab 1.3.0 foot_swing_height is fully
    # sensor-driven (no asset_cfg); only foot_clearance/foot_slip still carry an
    # asset_cfg whose site_names select the feet.
    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    # Body-specific configurations
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)

    # Discourage the friction-dependent foot skating observed on hardware while
    # retaining enough freedom for the later turning curriculum.
    cfg.rewards["foot_slip"].weight = -0.3
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.01

    cfg.rewards.pop("soft_landing", None)

    # Self-collision penalty: discourages legs from crashing into the trunk
    # battery holder (the self_collision_only-classed geoms on leg, leg_2,
    # battery_holder). With proper joint-range limits the policy can't actually
    # reach the body, but a positive signal here keeps it well clear.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )


    # Keep enough flight time and clearance to release a foot from carpet, but
    # do not pay for the exaggerated long/high swing found by stride_v3.
    cfg.rewards["air_time"].weight = 1.35 if stride_tuning else 1.25
    cfg.rewards["air_time"].params["command_threshold"] = 0.01
    cfg.rewards["air_time"].params["threshold_min"] = 0.160 if stride_tuning else 0.125
    cfg.rewards["air_time"].params["threshold_max"] = 0.340 if stride_tuning else 0.300

    cfg.rewards["body_ang_vel"].weight = -0.15
    cfg.rewards["angular_momentum"].weight = -0.04

    # Velocity tracking rewards
    cfg.rewards["track_linear_velocity"].weight = 4.0
    cfg.rewards["track_linear_velocity"].params["std"] = 0.18
    cfg.rewards["track_angular_velocity"].weight = 1.5
    cfg.rewards["track_angular_velocity"].params["std"] = 0.30

    # Smoothness is introduced only after locomotion starts, so exploration is
    # not frozen into the standstill solution.
    cfg.rewards["action_rate_l2"].weight = -0.03 if stride_tuning else -0.1
    cfg.rewards["leg_action_acceleration_l2"] = RewardTermCfg(
        func=microduck_mdp.leg_action_acceleration_l2,
        weight=0.0,
    )
    cfg.rewards["leg_joint_velocity_l2"] = RewardTermCfg(
        func=microduck_mdp.leg_joint_vel_l2,
        weight=0.0,
    )
    cfg.rewards["gait_symmetry"] = RewardTermCfg(
        func=microduck_mdp.gait_symmetry_penalty,
        weight=-0.35,
        params={"sensor_name": feet_ground_cfg.name},
    )
    # Directly shape effective step length at touchdown. Velocity tracking alone
    # permits a short-step/high-cadence solution; this event reward asks the new
    # support foot to land farther ahead/behind the other foot. It is gated to
    # straight locomotion and actual progress, so turning, standing and tapping
    # in place do not receive it. Weight starts at zero and ramps below.
    if stride_tuning:
        cfg.rewards["touchdown_stride"] = RewardTermCfg(
            func=microduck_mdp.touchdown_stride_reward,
            weight=0.0,
            params={
                "sensor_name": feet_ground_cfg.name,
                "command_name": "twist",
                "asset_cfg": SceneEntityCfg("robot", site_names=tuple(site_names)),
                "min_command_speed": 0.05,
                "min_air_time": 0.15,
                "base_target_m": 0.035,
                "speed_target_s": 0.13,
                "max_target_m": 0.070,
                "progress_speed_m_s": 0.08,
                "lateral_command_std": 0.10,
                "yaw_command_std": 0.30,
            },
        )
        # Dense companion to the touchdown event: during the latter half of a
        # single-foot swing, reward the airborne foot for passing the support
        # foot in the commanded direction. This supplies a gradient on many
        # control frames instead of only one touchdown frame, which is necessary
        # to overcome the dense velocity/pose stack's short-step optimum.
        cfg.rewards["swing_foot_lead"] = RewardTermCfg(
            func=microduck_mdp.swing_foot_lead_reward,
            weight=0.0,
            params={
                "sensor_name": feet_ground_cfg.name,
                "command_name": "twist",
                "asset_cfg": SceneEntityCfg("robot", site_names=tuple(site_names)),
                "min_command_speed": 0.05,
                "min_air_time": 0.08,
                "max_air_time": 0.52,
                "base_target_m": 0.035,
                "speed_target_s": 0.13,
                "max_target_m": 0.070,
                "progress_speed_m_s": 0.08,
                "lateral_command_std": 0.10,
                "yaw_command_std": 0.30,
            },
        )
        # Potential-based swing progress. Unlike swing_foot_lead, this pays
        # positive DELTA as soon as the foot starts advancing from behind the
        # support foot, so a from-scratch policy receives a learning signal
        # before it can already produce a full stride. Moving the foot back
        # produces an equal negative delta, preventing oscillation farming.
        cfg.rewards["swing_foot_progress"] = RewardTermCfg(
            func=microduck_mdp.swing_foot_progress_reward,
            weight=0.0,
            params={
                "sensor_name": feet_ground_cfg.name,
                "command_name": "twist",
                "asset_cfg": SceneEntityCfg("robot", site_names=tuple(site_names)),
                "min_command_speed": 0.05,
                "target_base_m": 0.035,
                "target_speed_s": 0.13,
                "target_max_m": 0.070,
                "progress_speed_m_s": 0.08,
                "lateral_command_std": 0.10,
                "yaw_command_std": 0.30,
                "max_normalized_delta": 0.30,
            },
        )
        cfg.rewards["short_swing"] = RewardTermCfg(
            func=microduck_mdp.short_swing_cost,
            weight=0.0,
            params={
                "sensor_name": feet_ground_cfg.name,
                "command_name": "twist",
                "minimum_swing_s": 0.20,
                "min_command_speed": 0.05,
            },
        )
    cfg.rewards["heading_hold"] = RewardTermCfg(
        func=microduck_mdp.heading_hold_reward,
        weight=2.0,
        params={"std": 0.14, "command_name": "twist", "command_std": 0.06},
    )
    # The Gaussian heading reward above is precise near zero error but loses
    # gradient after a sizeable veer.  This gated L1 companion keeps a constant
    # corrective gradient for straight forward commands without fighting turns.
    cfg.rewards["heading_error_l1"] = RewardTermCfg(
        func=microduck_mdp.heading_error_l1_penalty,
        weight=1.5,
        params={
            "command_name": "twist",
            "min_forward_speed": 0.05,
            "lateral_command_std": 0.05,
            "yaw_command_std": 0.06,
        },
    )
    cfg.rewards["action_over_limit"] = RewardTermCfg(
        func=microduck_mdp.action_over_limit_penalty,
        weight=-1.0,
        params={"action_name": "joint_pos", "overshoot": 0.10},
    )
    # Protect the real HD1910 linkages independently of action semantics.  A
    # low-kp servo can be commanded beyond its target without a large action
    # cost, so penalise measured qpos before a leg reaches a hard stop.
    cfg.rewards["leg_joint_limit_proximity"] = RewardTermCfg(
        func=microduck_mdp.joint_pos_limit_proximity,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
            ),
            "margin": 0.18,
        },
    )

    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.01
    cfg.rewards["foot_clearance"].params["target_height"] = 0.018 if stride_tuning else 0.014

    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.01
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.018 if stride_tuning else 0.014

    # NOTE: no neck-only action-rate term — the shared action_rate_l2 sums over
    # ALL action dims (neck included), and head_pose_tracking below gives the
    # 4 neck/head DOFs a position objective, so the neck is fully shaped.

    # Events
    # BAM (mjlab_frictionloss branch) writes per-env dof_frictionloss/dof_damping
    # every step; this no-op event registers those fields for per-world expansion.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )

    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = foot_frictions_geom_names
    # Carpet raises both traction and release drag.  The upper end deliberately
    # exceeds smooth-floor rubber so Flat learns a gait that does not depend on
    # easy sliding; the lower end keeps polished-floor coverage.
    cfg.events["foot_friction"].params["ranges"] = (0.6, 1.8)
    # Terminate environments that have gone numerically unstable (NaN physics).
    # MuJoCo can produce NaN joint positions on extreme contact impulses.
    # Terminating immediately resets to a valid state before NaN propagates
    # into the observation buffer and corrupts network weights.
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)

    # Velocity-based pushes for robustness training
    if ENABLE_VELOCITY_PUSHES and not play:
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=VELOCITY_PUSH_INTERVAL_S,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
    else:
        cfg.events.pop("push_robot", None)

    # Domain randomization — re-sampled per episode at reset. In mjlab 1.3.0 the
    # stock dr.* ops with operation="add"/"scale" read from the compile-time
    # default field each reset (Operation.uses_defaults=True), so they are
    # NON-accumulating natively — this upstream behavior replaces microduck's old
    # custom restore-then-add functions that worked around the accumulation footgun.
    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        # Randomize the CoM of the head assembly bodies (per-body fresh offset each reset).
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        # Randomize motor PD gains
        # Uses custom function that handles DelayedActuator
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        # Physics-consistent mass + inertia randomization via mjlab's pseudo_inertia:
        # alpha scales BOTH mass and inertia by e^(2*alpha) with the CoM unchanged
        # (so it does NOT conflict with randomize_com). alpha_range is derived from
        # the ±5% mass scale range: e^(2*alpha) ∈ [0.95, 1.05].
        # Replaces the old custom randomize_mass_and_inertia, which was a silent
        # no-op under mjlab 1.3.0 (direct per-env body_mass/body_inertia writes are
        # not expanded and collapse to a single shared value). Startup mode = fixed
        # per env for the whole run (standard for mass DR; no accumulation).
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        # Joint-friction DR under BAM: scales BAM's velocity-independent friction
        # budget (Coulomb + Stribeck + load) per-env via the FrictionDRBamActuator
        # friction_scale hook. MuJoCo's dof_frictionloss is zeroed under BAM, so the
        # stock dr.dof_frictionloss is a no-op — this is the BAM-native path.
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_JOINT_DAMPING_RANDOMIZATION:
        # Randomize joint damping (lubrication, temperature effects).
        # Custom non-accumulating scaler. NOTE: no-op under BAM (dof_damping
        # zeroed in edit_spec); only affects the XML position actuator.
        cfg.events["randomize_joint_damping"] = EventTermCfg(
            func=microduck_mdp.randomize_dof_field_scaled,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "field": "dof_damping",  # required by domain_randomization=True
                "scale_range": JOINT_DAMPING_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        # Randomize reflected rotor inertia (armature), microban-exact
        # (dr.joint_armature, scale, ±10%). Non-accumulating (uses_defaults). DOES
        # affect the BAM actuator — BAM sets dof_armature (~0.0018), it isn't zeroed.
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # IMU orientation randomization (mounting error) is applied at the OBSERVATION
    # level below (per-env constant rotation of projected_gravity + base_ang_vel).
    # The old event-based randomize_imu_orientation wrote site_quat, which under
    # mjlab 1.3.0 is neither per-env expanded nor read by these obs — a no-op.

    # Base orientation randomization (forces reactive behavior)
    if ENABLE_BASE_ORIENTATION_RANDOMIZATION:
        cfg.events["randomize_base_orientation"] = EventTermCfg(
            func=microduck_mdp.randomize_base_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_pitch_deg": BASE_ORIENTATION_MAX_PITCH_DEG,
                "max_roll_deg": BASE_ORIENTATION_MAX_ROLL_DEG,
            },
        )

    # Observations
    del cfg.observations["actor"].terms["base_lin_vel"]
    # mjlab 1.3.0 adds a height_scan term (terrain ray scan) to both groups by
    # default. The microduck has no such body-mounted terrain sensor for the
    # policy, so drop it from both (mirrors microban).
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # Add base_lin_vel to critic only (privileged information)
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )

    # Determine gravity/accelerometer term name based on flag
    gravity_term_name = "projected_gravity" if USE_PROJECTED_GRAVITY else "raw_accelerometer"

    # Replace projected_gravity with raw_accelerometer if flag is False
    if not USE_PROJECTED_GRAVITY:
        # Remove projected_gravity and add raw_accelerometer
        del cfg.observations["actor"].terms["projected_gravity"]
        cfg.observations["actor"].terms["raw_accelerometer"] = ObservationTermCfg(
            func=microduck_mdp.raw_accelerometer,
            scale=1.0,
        )

    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1  # was 3 (=60 ms worst case); real dxl IMU path is fast — ±20 ms envelope (2026-07 audit)
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64

    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1  # was 3 (=60 ms worst case); real dxl IMU path is fast — ±20 ms envelope (2026-07 audit)
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # The critic's sensor-derived terms are the one obs path `nan_state` cannot
    # protect (it checks joint + root state; these read raycast/contact sensor
    # data, which MuJoCo can return non-finite for while the state is still
    # clean). A single NaN here kills the whole run via rsl_rl's check_nan —
    # that is the 2026-08-21 Velocity2-Rough-Backlash crash. Critic-only, so
    # sanitizing costs the policy nothing.
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_height", microduck_mdp.foot_height_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    # Observation noise configuration (edit these values as needed)
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03) # was 0.2
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)  # was 0.15
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)  # was 0.05
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.08, n_max=0.08)

    # IMU mounting-misalignment DR (per-env constant rotation of the IMU-derived
    # observations). Applied to the ACTOR only (the policy sees a slightly rotated
    # IMU frame, like a real mounting error); the critic keeps the true values.
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        if USE_PROJECTED_GRAVITY:
            g = cfg.observations["actor"].terms[gravity_term_name]
            g.func = microduck_mdp.projected_gravity_imu_misaligned
            g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # 1-ctrl-step lag on joint_vel: the Dynamixel firmware computes
    # present_velocity via a moving-average over the previous position-sample
    # window, so the value the policy actually reads is ~1 control period old.
    # Matches reality and stops the policy relying on instantaneous qdot feedback.
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # Exclude passive_* joints (jaw linkage) from joint_pos/vel obs so the
    # observation dim matches the action dim (14) instead of the raw articulation (16).
    # Deepcopy each joint_pos/joint_vel term first — actor and critic share the
    # same term objects/params dicts from the base template, so mutating one would
    # leak into the other (e.g. the encoder-bias `biased` flag below).
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    # Encoder-bias DR: the base template samples a per-env constant joint-encoder
    # offset (startup event "encoder_bias"), but joint_pos_rel ignores it unless
    # biased=True. Feed the biased joint pos to the ACTOR only (what the real
    # encoders report); the critic keeps the true joint pos (privileged).
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Commands — deepcopy to avoid shared-state corruption from other env cfgs
    # (make_velocity_env_cfg() returns objects with shared mutable references;
    # standup/ground_pick envs mutate commands["twist"] in place, zeroing ranges)
    command: UniformVelocityCommandCfg = deepcopy(cfg.commands["twist"])
    cfg.commands["twist"] = command
    command.rel_standing_envs = 0.05
    command.rel_heading_envs = 0.0
    # Bootstrap around the measured 0.1--0.2 m/s operating region. The
    # curriculum below expands the envelope only after the slow gait exists.
    command.ranges.lin_vel_x = (-0.18, 0.18)
    command.ranges.lin_vel_y = (-0.04, 0.04)
    command.ranges.ang_vel_z = (-0.20, 0.20)
    command.viz.z_offset = 0.5
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))
    # Explicit turn-in-place bucket (see TURN_IN_PLACE_FRACTION above).
    cfg.commands["twist"].rel_turn_in_place_envs = TURN_IN_PLACE_FRACTION
    cfg.commands["twist"].rel_slow_forward_envs = (
        SLOW_FORWARD_FRACTION if stride_tuning else 0.40
    )
    cfg.commands["twist"].slow_forward_range = (
        (0.10, 0.30) if stride_tuning else (0.06, 0.20)
    )

    # Head pose command (4D deltas from HOME, in joint order:
    #   neck_pitch, head_pitch, head_yaw, head_roll). Tracked as a primary
    # reward — see "head_pose_tracking" added below. Initial ranges are small
    # non-zero so input neurons stay alive from step 0; curriculum widens them.
    # Per-joint final caps reflect each joint's mechanically reachable delta
    # from HOME (XML limits minus HOME offset, with ~10% safety margin):
    #   neck_pitch / head_pitch: ±1.10 rad (limit ±π/2 with the verified HOME)
    #   head_yaw                : ±1.40 rad (limit ±π/2 with HOME=0)
    #   head_roll               : ±0.31 rad (limit ±20°)
    # Initial ranges are small non-zero so input neurons stay alive from step 0.
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        zero_command_prob=HEAD_ZERO_COMMAND_PROB,
        ranges=(
            (-0.05, 0.05),    # neck_pitch
            (-0.05, 0.05),    # head_pitch
            (-0.07, 0.07),    # head_yaw
            (-0.015, 0.015),  # head_roll (tighter — much smaller mechanical range)
        ),
    )
    # Body pose command (6D delta from nominal standing: [x, y, z, roll, pitch, yaw]).
    # Vel env carries this slot for runtime obs-shape parity; tracked at a tiny
    # weight to keep the input neurons alive but not steer the policy. The
    # standup env raises the weight + widens the ranges.
    cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.005, 0.005),  # x (m)
            (-0.005, 0.005),  # y (m)
            (-0.005, 0.005),  # z (m)
            (-0.05, 0.05),    # roll (rad)
            (-0.05, 0.05),    # pitch (rad)
            (-0.05, 0.05),    # yaw (rad)
        ),
    )

    # Append head + body command obs terms to both policy and critic groups.
    # Order matters for the runtime obs layout: [twist(3), head_pose(4), body_pose(6)].
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "body_pose"},
        )

    # === Pose tracking rewards ===
    # head_pose: primary objective in vel env — the whole point of the rewrite.
    # std=0.5 with per-joint Gaussian (see head_pose_tracking in mdp.py): at the
    # full ±1.0 rad command, a non-tracking policy still sees per-joint reward
    # exp(-(1/0.5)²)=exp(-4)≈0.018 — a small but non-zero gradient — so the
    # curriculum widening doesn't kill the signal. Final reward is the mean
    # over 4 joints, so partial tracking is partial reward (no all-or-nothing).
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=2.0,
        params={"command_name": "head_pose", "std": 0.5},
    )
    # body_pose: infra kept intact but DISABLED (weight 0) — the obs slot and
    # command stay alive for envs that raise the weight (standup).
    cfg.rewards["body_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.body_pose_tracking_6d,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "nominal_height": 0.095,
            "xy_std": 0.05,
            "z_std": 0.02,
            "angle_std": math.radians(15),
        },
    )

    # Head droop fix (2026-08-20). The head walks pitched ~15° down (measured:
    # run ww1g2198 head_pose_tracking 1.544/2.0 → 14.6° mean joint error).
    # DO NOT fix this by tightening head_pose_tracking's std: run 5yay13u4 tried
    # fine_std=0.1 and the policy stopped walking entirely by iter 300 (air_time
    # 1.01 → 0.02, peak foot height 15 mm → 2 mm, entropy collapsed 10.9 → 1.9).
    # An instantaneous tight tolerance taxes walking 0.77/step — 76% of the whole
    # air_time reward — and is UNESCAPABLE, since a 280 g head (38% of robot
    # mass) must oscillate while stepping. Standing still scored higher, so it
    # stood still.
    # The DC bias, unlike the oscillation, IS escapable (bias the neck command up
    # to cancel gravity sag), so price only that: L1 on a 1 s EMA of the error.
    # At the optimum this costs a walking policy nothing.
    cfg.rewards["head_pose_bias"] = RewardTermCfg(
        func=microduck_mdp.head_pose_bias_penalty,
        weight=0.0,  # ramped by the head_pose_bias_weight curriculum below
        params={
            "command_name": "head_pose",
            "tau_s": 1.0,
            "component_indices": (0, 1),
        },
    )

    # Terrain
    if not rough:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    else:
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = MICRODUCK_ROUGH_TERRAINS_CFG

        # Soften terrain box contacts: adjacent boxes at different heights create
        # hard edges that destabilise the contact solver and produce NaN forces.
        cfg.scene.spec_fn = _soften_terrain_contacts

        # The velocity env default nconmax=35 is tight for rough terrain: when the
        # robot falls and multiple body links hit multiple boxes simultaneously,
        # contacts overflow → some are silently dropped → sudden decompression → NaN.
        cfg.sim.nconmax = 200   # was 35

        # The velocity env uses only 10 solver iterations (vs the default 100),
        # which is too few to resolve edge contacts on rough box terrain.
        # Tripling iterations significantly reduces contact resolution failures
        # with a modest compute cost on GPU (MJWarp parallelises across envs).
        cfg.sim.mujoco.iterations = 30    # was 10
        cfg.sim.mujoco.ls_iterations = 50  # was 20

        if play:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5

    # Action-rate weight ramp: discover the gait first, then suppress the target
    # jumps measured on hardware (p95 4.15 rad/s, max 46 rad/s at vx=0.2).
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": (
                [
                    {"step": 0, "weight": -0.03},
                    {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.08},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.20},
                    {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -0.40},
                    {"step": 2500 * NUM_STEPS_PER_ENV, "weight": -1.10},
                ]
                if stride_tuning
                else [
                    {"step": 0, "weight": -0.05},
                    {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.15},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.4},
                    {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -0.8},
                    {"step": 2500 * NUM_STEPS_PER_ENV, "weight": -1.2},
                ]
            ),
        },
    )

    # Gradually increase standing env fraction after walking is established
    cfg.curriculum["standing_envs"] = CurriculumTermCfg(
        func=microduck_mdp.standing_envs_curriculum,
        params={
            "command_name": "twist",
            "standing_stages": [
                {"step": 0, "rel_standing_envs": 0.05},
                {"step": 1000 * NUM_STEPS_PER_ENV, "rel_standing_envs": 0.08},
                {"step": 2500 * NUM_STEPS_PER_ENV, "rel_standing_envs": 0.10},
            ],
        },
    )

    cfg.curriculum["command_velocity"] = CurriculumTermCfg(
        func=microduck_mdp.velocity_command_ranges_curriculum,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {"step": 0, "lin_vel_x_range": 0.18, "lin_vel_y_range": 0.04, "ang_vel_range": 0.20},
                {"step": 750 * NUM_STEPS_PER_ENV, "lin_vel_x_range": 0.24, "lin_vel_y_range": 0.07, "ang_vel_range": 0.35},
                {"step": 1500 * NUM_STEPS_PER_ENV, "lin_vel_x_range": 0.30, "lin_vel_y_range": 0.12, "ang_vel_range": 0.55},
                {"step": 3000 * NUM_STEPS_PER_ENV, "lin_vel_x_range": 0.40, "lin_vel_y_range": 0.20, "ang_vel_range": 0.80},
            ],
        },
    )

    cfg.curriculum["linear_velocity_std"] = CurriculumTermCfg(
        func=microduck_mdp.velocity_tracking_std_curriculum,
        params={
            "reward_name": "track_linear_velocity",
            "std_stages": [
                {"step": 0, "std": 0.18},
                {"step": 1000 * NUM_STEPS_PER_ENV, "std": 0.15},
                {"step": 2500 * NUM_STEPS_PER_ENV, "std": 0.12},
            ],
        },
    )

    cfg.curriculum["leg_action_acceleration_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "leg_action_acceleration_l2",
            "weight_stages": (
                [
                    {"step": 0, "weight": 0.0},
                    {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.03},
                    {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -0.06},
                    {"step": 2500 * NUM_STEPS_PER_ENV, "weight": -0.12},
                ]
                if stride_tuning
                else [
                    {"step": 0, "weight": 0.0},
                    {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.02},
                    {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -0.05},
                    {"step": 2500 * NUM_STEPS_PER_ENV, "weight": -0.10},
                ]
            ),
        },
    )

    # Directly discourage fast physical leg motion.  Unlike action-rate this
    # also sees the servo/ground response, so it slows a mature gait without
    # requiring a smaller command or changing the 61D observation contract.
    cfg.curriculum["leg_joint_velocity_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "leg_joint_velocity_l2",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.003},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -0.006},
                {"step": 2500 * NUM_STEPS_PER_ENV, "weight": -0.010},
            ],
        },
    )

    # From-scratch runs introduce the sparse touchdown signal progressively.
    # HOME and the observation/action contract are unchanged, so a mature
    # same-task checkpoint may also resume here; restored step counters select
    # the final smoothing weights immediately, which is intended for fine-tune.
    if stride_tuning:
        cfg.curriculum["touchdown_stride_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "touchdown_stride",
                "weight_stages": [
                    {"step": 0, "weight": 0.0},
                    {"step": 250 * NUM_STEPS_PER_ENV, "weight": 2.0},
                    {"step": 750 * NUM_STEPS_PER_ENV, "weight": 4.0},
                    {"step": 1500 * NUM_STEPS_PER_ENV, "weight": 6.0},
                ],
            },
        )
        cfg.curriculum["swing_foot_lead_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "swing_foot_lead",
                "weight_stages": [
                    {"step": 0, "weight": 0.0},
                    {"step": 250 * NUM_STEPS_PER_ENV, "weight": 0.5},
                    {"step": 750 * NUM_STEPS_PER_ENV, "weight": 1.0},
                    {"step": 1500 * NUM_STEPS_PER_ENV, "weight": 2.0},
                ],
            },
        )
        cfg.curriculum["swing_foot_progress_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "swing_foot_progress",
                "weight_stages": [
                    {"step": 0, "weight": 0.0},
                    {"step": 150 * NUM_STEPS_PER_ENV, "weight": 1.0},
                    {"step": 500 * NUM_STEPS_PER_ENV, "weight": 2.5},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "weight": 4.0},
                ],
            },
        )
        cfg.curriculum["short_swing_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "short_swing",
                "weight_stages": [
                    {"step": 0, "weight": 0.0},
                    {"step": 750 * NUM_STEPS_PER_ENV, "weight": -1.0},
                    {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -2.0},
                    {"step": 2500 * NUM_STEPS_PER_ENV, "weight": -3.0},
                ],
            },
        )

    if ENABLE_VELOCITY_PUSHES and not play:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}},
                    {"step": 2000 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)}},
                    {"step": 3500 * NUM_STEPS_PER_ENV, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )
    else:
        cfg.curriculum.pop("push_magnitude", None)

    # Head pose command range curriculum — per-joint, scaled to each joint's
    # reachable delta from HOME (with ~10% margin from XML limits). Same 5-stage
    # shape as before (5% → 15% → 35% → 65% → 100% of each joint's final cap).
    # neck/head pitch final ±1.10 rad, head_yaw ±1.40, head_roll ±0.31.
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                # step,                ranges = ((neck_pitch), (head_pitch), (head_yaw),  (head_roll))
                {"step": 0,         "ranges": ((-0.05, 0.05),  (-0.05, 0.05),  (-0.07, 0.07),  (-0.015, 0.015))},
                {"step": 500 * 24,  "ranges": ((-0.17, 0.17),  (-0.17, 0.17),  (-0.21, 0.21),  (-0.047, 0.047))},
                {"step": 1000 * 24, "ranges": ((-0.39, 0.39),  (-0.39, 0.39),  (-0.49, 0.49),  (-0.11, 0.11))},
                {"step": 1500 * 24, "ranges": ((-0.72, 0.72),  (-0.72, 0.72),  (-0.91, 0.91),  (-0.20, 0.20))},
                {"step": 2000 * 24, "ranges": ((-1.10, 1.10),  (-1.10, 1.10),  (-1.40, 1.40),  (-0.31, 0.31))},
            ],
        },
    )

    # Body pose command range curriculum: stay small in vel env. Standup env
    # overrides this curriculum with wide ranges + heavy reward weight.
    cfg.curriculum["body_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "body_pose",
            "range_stages": [
                {"step": 0, "ranges": (
                    (-0.005, 0.005),  # x (m)
                    (-0.005, 0.005),  # y (m)
                    (-0.005, 0.005),  # z (m)
                    (-0.05, 0.05),    # roll
                    (-0.05, 0.05),    # pitch
                    (-0.05, 0.05),    # yaw
                )},
            ],
        },
    )

    # CoM randomization range curriculum - start small, ramp up
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    # Capped at ±15 mm (2026-07 audit): the previous ramp to ±30 mm
                    # exceeded the foot support polygon (heel is only 20 mm behind
                    # the ankle) — the randomized CoM could sit entirely outside
                    # support, forcing a wide/fast hyper-reactive gait and making
                    # BACKWARD balance untrainable. Regression timeline matched the
                    # ramp increases: 0.015 → 0.02 → 0.03 as policies got worse.
                    {"step": 0, "range": 0.002},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "range": 0.004},
                    {"step": 2500 * NUM_STEPS_PER_ENV, "range": 0.008},
                ],
            },
        )

    # Head CoM randomization range curriculum - start small, ramp up
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    # Capped at ±10 mm (2026-07 audit — same over-conservatism
                    # concern as trunk CoM; head is a large lever arm).
                    {"step": 0, "range": 0.002},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "range": 0.0035},
                    {"step": 2500 * NUM_STEPS_PER_ENV, "range": 0.005},
                ],
            },
        )

    # Disable default curriculum
    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # head_pose_bias ramp: OFF until iter 600, then 1.5 → 4.5 by iter 1500.
    # Held at 0 early because a posture-precision term is a distraction before
    # a gait exists. At the final weight 4.5, a 15° residual bias costs
    # 1.18/step and a 2° bias costs 0.16/step.
    cfg.curriculum["head_pose_bias_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_pose_bias",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 600 * NUM_STEPS_PER_ENV, "weight": 1.5},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": 3.0},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": 4.5},
            ],
        },
    )

    return cfg


MicroduckRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    # New namespace prevents an incompatible pre-stability checkpoint from
    # being resumed accidentally after the mass/reward/command changes above.
    # Keep the experiment root aligned with the deployed stride_v3 lineage so
    # --agent.load-run resolves the real source checkpoint during fine-tuning.
    experiment_name="velocity_hd1910_stride_v3",
    run_name="straight_head_finetune_v5",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=6_000,
)
