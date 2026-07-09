"""Table-tennis latent-control task configuration."""

from __future__ import annotations

import math

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.pingpong import mdp
from mjlab.tasks.pingpong.bounce import (
  PINGPONG_POST_BOUNCE_HORIZONTAL_SCALE,
  PINGPONG_POST_BOUNCE_VERTICAL_SCALE,
)
from mjlab.tasks.pingpong.mdp.ball_providers import (
  TableTennisFeederCfg,
  TrajectoryCheckCfg,
)
from mjlab.tasks.pingpong.pace_geometry import G1_PACE_GEOMETRY
from mjlab.tasks.pingpong.scene import (
  BALL_CENTER_TABLE_Z,
  NET_TOP_Z,
  NET_X,
  TABLE_HALF_LENGTH,
  TABLE_HALF_WIDTH,
  get_pingpong_ball_cfg,
  get_pingpong_table_cfg,
  get_pingpong_terrain_cfg,
)
from mjlab.tasks.tennis.mdp import FrozenDecoderLatentJointPositionActionCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

_ROBOT_CFG = SceneEntityCfg("robot", joint_names=(".*",))
_PADDLE_CFG = SceneEntityCfg("robot", site_names=("pingpong_paddle_center",))
_PADDLE_GEOM_CFG = SceneEntityCfg("robot", geom_names=("pingpong_paddle_collision",))
_BALL_CFG = SceneEntityCfg("ball")
_TABLE_CFG = SceneEntityCfg("table")
_PADDLE_BALL_SENSOR = "paddle_ball_contact"
_BALL_NET_SENSOR = "pingpong_ball_net_contact"
_ROBOT_TABLE_SENSOR = "robot_table_contact"
_ROBOT_BALL_SENSOR = "robot_ball_contact"
PACE_FOOT_CONTACT_SENSOR = "pace_foot_contact"

PADDLE_BALL_PAIR_NAME = "pingpong_paddle_ball_contact_pair"
PADDLE_BALL_PAIR_GEOM1 = "ball/pingpong_ball"
PADDLE_BALL_PAIR_GEOM2 = "robot/pingpong_paddle_collision"
PADDLE_BALL_PAIR_CONDIM = 3
PADDLE_BALL_PAIR_FRICTION = (0.08, 0.002, 0.0001)
PADDLE_BALL_PAIR_SOLREF = (0.011, 0.40)
PADDLE_BALL_PAIR_SOLIMP = (0.93, 0.98, 0.001, 0.5, 2.0)
PADDLE_BALL_PAIR_MARGIN = 0.010
_PADDLE_BALL_PAIR_FRICTION_FULL = PADDLE_BALL_PAIR_FRICTION + (0.0001, 0.0001)

ROBOT_RESET_X_RANGE = (TABLE_HALF_LENGTH + 0.36, TABLE_HALF_LENGTH + 0.58)
ROBOT_RESET_Y_RANGE = (-0.18, 0.18)
ROBOT_RESET_YAW = math.pi

# Provider target x ranges are robot-side baseline margins when
# target_x_range_mode="self_baseline_margin"; y ranges are half-field fractions.
BALL_TARGET_INITIAL_X_RANGE = (0.30, 0.12)
BALL_TARGET_INITIAL_Y_RANGE = (-0.13, 0.13)
BALL_TARGET_X_RANGE = (0.50, 0.12)
BALL_TARGET_Y_RANGE = (-0.70, 0.70)
BALL_TARGET_CURRICULUM_SUCCESS_THRESHOLD = 0.75
BALL_TARGET_CURRICULUM_WINDOW = 50
BALL_TARGET_CURRICULUM_STAGES = 6
ACTION_REGULARIZATION_CURRICULUM_SUCCESS_THRESHOLD = 0.80
ACTION_REGULARIZATION_CURRICULUM_WINDOW = 50
ACTION_REGULARIZATION_CURRICULUM_STAGE_WEIGHTS = (
  {
    "latent_action_rate_l2": -0.005,
    "joint_torques_l2": -2.0e-5,
    "joint_acc_l2": -2.0e-6,
    "fall_penalty": -200.0,
    "flat_orientation_l2": 0.0,
  },
  {
    "latent_action_rate_l2": -0.0065,
    "joint_torques_l2": -3.0e-5,
    "joint_acc_l2": -3.0e-6,
    "fall_penalty": -300.0,
    "flat_orientation_l2": -0.7,
  },
  {
    "latent_action_rate_l2": -0.008,
    "joint_torques_l2": -4.0e-5,
    "joint_acc_l2": -4.0e-6,
    "fall_penalty": -400.0,
    "flat_orientation_l2": -1.3,
  },
  {
    "latent_action_rate_l2": -0.01,
    "joint_torques_l2": -5.0e-5,
    "joint_acc_l2": -5.0e-6,
    "fall_penalty": -500.0,
    "flat_orientation_l2": -2.0,
  },
)
CROSS_LOOSE_REGULARIZATION_WEIGHTS = {
  "latent_action_rate_l2": -0.0025,
  "low_level_action_rate_l2": 0.0,
  "joint_torques_l2": -2.0e-5,
  "joint_acc_l2": -1.0e-6,
  "fall_penalty": -300.0,
  "flat_orientation_l2": -0.5,
}
CROSS_HIT_POINT_WEIGHT = 10.0
CROSS_ROBOT_BALL_CONTACT_WEIGHT = -75.0
CROSS_POST_HIT_X_PROGRESS_WEIGHT = 60.0
CROSS_POST_HIT_BALL_VELOCITY_DIRECTION_WEIGHT = 120.0
CROSS_LAND_OPPONENT_WEIGHT = 1200.0
CROSS_STRIKE_QUALITY_REWARD_WEIGHTS: dict[str, float] = {
  "strike_pred_net_clearance": 40.0,
  "strike_pred_landing_inside": 80.0,
  "strike_post_hit_speed": 30.0,
}
CROSS_IMPACT_REWARD_WEIGHTS: dict[str, float] = {}
PACE_TASK_REWARD_WEIGHTS: dict[str, float] = {
  "pace_contact": 150.0,
  "pace_future_ee_target": 8.0,
  "pace_future_paddle_height_target": 4.0,
  "pace_future_body_target": 5.0,
  "pace_future_base_vel_target": 5.0,
  "pace_future_landing_distance": 60.0,
  "pace_future_pass_net": 100.0,
  "pace_table_success": 100.0,
  "pace_forehand_paddle_offset": 3.0,
  "pace_forehand_elbow_extension": 2.0,
  "pace_step_air_time": 0.5,
}
PACE_TARGET_BASE_OFFSET_XY = G1_PACE_GEOMETRY.target_base_offset_xy
PACE_NATURAL_HIT_X = G1_PACE_GEOMETRY.natural_hit_x
PACE_TARGET_ROOT_HEIGHT = G1_PACE_GEOMETRY.target_root_height
PACE_TARGET_BASE_VEL_GAIN = G1_PACE_GEOMETRY.target_base_vel_gain
PACE_TARGET_BASE_VEL_MAX = G1_PACE_GEOMETRY.target_base_vel_max
PACE_FOREHAND_PADDLE_OFFSET = G1_PACE_GEOMETRY.forehand_paddle_offset
PACE_FOREHAND_PADDLE_OFFSET_STD = G1_PACE_GEOMETRY.forehand_paddle_offset_std
PACE_FOREHAND_ELBOW_TARGET_RATIO = G1_PACE_GEOMETRY.forehand_elbow_target_ratio
PACE_FOOT_GEOM_NAMES = G1_PACE_GEOMETRY.foot_geom_names
PACE_BAD_ORIENTATION_LIMIT = G1_PACE_GEOMETRY.bad_orientation_limit
PACE_ROOT_HEIGHT_MINIMUM = G1_PACE_GEOMETRY.root_height_minimum

OUT_X_LIMITS = (-TABLE_HALF_LENGTH - 0.75, TABLE_HALF_LENGTH + 1.10)
OUT_Y_LIMITS = (-TABLE_HALF_WIDTH - 0.50, TABLE_HALF_WIDTH + 0.50)
OUT_Z_LIMITS = (0.05, 2.5)
SELF_X_LIMITS = (NET_X, TABLE_HALF_LENGTH)
OPPONENT_X_LIMITS = (-TABLE_HALF_LENGTH, NET_X)
TABLE_Y_LIMITS = (-TABLE_HALF_WIDTH, TABLE_HALF_WIDTH)
HIT_FORCE_THRESHOLD = 0.05
BOUNCE_Z_TOLERANCE = 0.055
IMPACT_WINDOW_DISTANCE = 0.45
IMPACT_FOLLOWTHROUGH_STEPS = 6
IMPACT_TARGET_X = -0.5 * TABLE_HALF_LENGTH
IMPACT_TARGET_Y = 0.0
IMPACT_TARGET_Z = NET_TOP_Z + 0.20

HIT_POINT_HEIGHT_OFFSET = 0.02
HIT_POINT_MAX_HORIZON = 1.2
DECODER_STATE_TERMS = (
  "base_lin_vel",
  "base_ang_vel",
  "joint_pos",
  "joint_vel",
  "actions",
)


def add_pingpong_paddle_ball_contact_pair(spec: mujoco.MjSpec) -> None:
  """Add the pingpong-only explicit paddle-face/ball contact pair."""
  spec.add_pair(
    name=PADDLE_BALL_PAIR_NAME,
    geomname1=PADDLE_BALL_PAIR_GEOM1,
    geomname2=PADDLE_BALL_PAIR_GEOM2,
    condim=PADDLE_BALL_PAIR_CONDIM,
    solref=PADDLE_BALL_PAIR_SOLREF,
    solimp=PADDLE_BALL_PAIR_SOLIMP,
    margin=PADDLE_BALL_PAIR_MARGIN,
    friction=_PADDLE_BALL_PAIR_FRICTION_FULL,
  )


def _state_params() -> dict[str, object]:
  return {
    "paddle_sensor_name": _PADDLE_BALL_SENSOR,
    "net_sensor_name": _BALL_NET_SENSOR,
    "body_ball_sensor_name": _ROBOT_BALL_SENSOR,
    "ball_cfg": _BALL_CFG,
    "paddle_cfg": _PADDLE_CFG,
    "paddle_geom_cfg": _PADDLE_GEOM_CFG,
    "robot_cfg": _ROBOT_CFG,
    "force_threshold": HIT_FORCE_THRESHOLD,
    "table_z": BALL_CENTER_TABLE_Z,
    "net_x": NET_X,
    "net_top_z": NET_TOP_Z,
    "gravity": 9.81,
    "self_x_limits": SELF_X_LIMITS,
    "opponent_x_limits": OPPONENT_X_LIMITS,
    "table_y_limits": TABLE_Y_LIMITS,
    "x_limits": OUT_X_LIMITS,
    "y_limits": OUT_Y_LIMITS,
    "z_limits": OUT_Z_LIMITS,
    "bounce_z_tolerance": BOUNCE_Z_TOLERANCE,
    "impact_window_distance": IMPACT_WINDOW_DISTANCE,
    "impact_followthrough_steps": IMPACT_FOLLOWTHROUGH_STEPS,
    "impact_target_x": IMPACT_TARGET_X,
    "impact_target_y": IMPACT_TARGET_Y,
    "impact_target_z": IMPACT_TARGET_Z,
  }


def _pace_state_params() -> dict[str, object]:
  return {
    **_state_params(),
    "target_base_offset_xy": PACE_TARGET_BASE_OFFSET_XY,
    "natural_hit_x": PACE_NATURAL_HIT_X,
    "target_root_height": PACE_TARGET_ROOT_HEIGHT,
    "target_base_vel_gain": PACE_TARGET_BASE_VEL_GAIN,
    "target_base_vel_max": PACE_TARGET_BASE_VEL_MAX,
  }


def _ball_provider_cfg() -> TableTennisFeederCfg:
  return TableTennisFeederCfg(
    ball_cfg=_BALL_CFG,
    spawn_x_range=(0.20, 0.20),
    spawn_x_range_mode="opponent_side_margin",
    spawn_y_range=(-0.55, 0.55),
    spawn_y_range_mode="field_fraction",
    target_x_range=BALL_TARGET_INITIAL_X_RANGE,
    target_x_range_mode="self_baseline_margin",
    target_y_range=BALL_TARGET_INITIAL_Y_RANGE,
    target_y_range_mode="field_fraction",
    vz_std=0.35,
    vz_max=3.4,
    post_bounce_horizontal_scale=PINGPONG_POST_BOUNCE_HORIZONTAL_SCALE,
    post_bounce_vertical_scale=PINGPONG_POST_BOUNCE_VERTICAL_SCALE,
    check=TrajectoryCheckCfg(
      require_edge_crossing=True,
      require_second_bounce_outside_self_half=True,
      net_clearance=0.06,
      edge_clearance=0.02,
      second_bounce_outside_margin=0.06,
      flight_time_range=(0.32, 0.75),
      vx_range=(2.0, 8.0),
      vy_abs_max=2.0,
    ),
  )


def _pace_observation_terms() -> dict[str, ObservationGroupCfg]:
  state_params = _pace_state_params()
  proprio_actor = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      params={"asset_cfg": _ROBOT_CFG},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": _ROBOT_CFG, "biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": _ROBOT_CFG},
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(
      func=mdp.last_action,
      params={"action_name": "joint_pos"},
    ),
    "ball_pos": ObservationTermCfg(
      func=mdp.pace_ball_position_table,
      params={"ball_cfg": _BALL_CFG},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "robot_pos": ObservationTermCfg(
      func=mdp.pace_robot_position_table,
      params={"robot_cfg": _ROBOT_CFG},
    ),
    "ball_prediction": ObservationTermCfg(
      func=mdp.pace_ball_prediction_table,
      params=dict(state_params),
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "rel_target_base_xy": ObservationTermCfg(
      func=mdp.pace_relative_target_base_xy,
      params=dict(state_params),
    ),
    "heading": ObservationTermCfg(
      func=mdp.pace_heading,
      params={"robot_cfg": _ROBOT_CFG},
    ),
  }
  critic_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": _ROBOT_CFG, "biased": False},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "actions": ObservationTermCfg(
      func=mdp.last_action,
      params={"action_name": "joint_pos"},
    ),
    "ball_pos": ObservationTermCfg(
      func=mdp.pace_ball_position_table,
      params={"ball_cfg": _BALL_CFG},
    ),
    "robot_pos": ObservationTermCfg(
      func=mdp.pace_robot_position_table,
      params={"robot_cfg": _ROBOT_CFG},
    ),
    "ball_velocity": ObservationTermCfg(
      func=mdp.ball_velocity_b,
      params={"ball_cfg": _BALL_CFG, "robot_cfg": _ROBOT_CFG},
    ),
    "ball_prediction": ObservationTermCfg(
      func=mdp.pace_ball_prediction_table,
      params=dict(state_params),
    ),
    "future_ball_pose": ObservationTermCfg(
      func=mdp.pace_future_ball_pose_table,
      params=dict(state_params),
    ),
    "paddle_touch_point": ObservationTermCfg(
      func=mdp.pace_paddle_touch_point_table,
      params={"paddle_cfg": _PADDLE_CFG},
    ),
    "robot_future_delta": ObservationTermCfg(
      func=mdp.pace_robot_future_delta,
      params=dict(state_params),
    ),
    "future_time": ObservationTermCfg(
      func=mdp.pace_future_time,
      params=dict(state_params),
    ),
    "rally_flags": ObservationTermCfg(
      func=mdp.pace_rally_flags,
      params=dict(state_params),
    ),
  }
  return {
    "actor": ObservationGroupCfg(
      proprio_actor,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }


def make_pingpong_latent_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the post-bounce legal-hit table-tennis task."""
  proprio_actor = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": _ROBOT_CFG, "biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": _ROBOT_CFG},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "actions": ObservationTermCfg(
      func=mdp.low_level_action,
      params={"action_name": "latent_joint_pos"},
    ),
  }
  actor_terms = dict(proprio_actor)
  actor_terms["ball_pos_window"] = ObservationTermCfg(
    func=mdp.ball_position_b,
    params={
      "ball_cfg": _BALL_CFG,
      "robot_cfg": _ROBOT_CFG,
    },
    noise=Unoise(n_min=-0.01, n_max=0.01),
    history_length=10,
    flatten_history_dim=True,
  )
  actor_terms["predicted_hit_point"] = ObservationTermCfg(
    func=mdp.ball_predicted_edge_hit_point_b,
    params={
      "ball_cfg": _BALL_CFG,
      "robot_cfg": _ROBOT_CFG,
      "hit_height_offset": HIT_POINT_HEIGHT_OFFSET,
      "max_horizon": HIT_POINT_MAX_HORIZON,
    },
    noise=Unoise(n_min=-0.01, n_max=0.01),
  )
  critic_terms = {
    name: ObservationTermCfg(func=t.func, params=dict(t.params))
    for name, t in proprio_actor.items()
  }
  critic_terms.update(
    {
      "paddle_to_ball": ObservationTermCfg(
        func=mdp.racket_to_ball_b,
        params={
          "racket_cfg": _PADDLE_CFG,
          "ball_cfg": _BALL_CFG,
          "robot_cfg": _ROBOT_CFG,
        },
      ),
      "ball_velocity": ObservationTermCfg(
        func=mdp.ball_velocity_b,
        params={"ball_cfg": _BALL_CFG, "robot_cfg": _ROBOT_CFG},
      ),
      "paddle_velocity": ObservationTermCfg(
        func=mdp.racket_velocity_b,
        params={"racket_cfg": _PADDLE_CFG, "robot_cfg": _ROBOT_CFG},
      ),
      "ball_predicted_landing": ObservationTermCfg(
        func=mdp.ball_predicted_landing_b,
        params={
          "ball_cfg": _BALL_CFG,
          "robot_cfg": _ROBOT_CFG,
          "ground_z": BALL_CENTER_TABLE_Z,
          "max_horizon": HIT_POINT_MAX_HORIZON,
        },
      ),
      "ball_predicted_hit_point": ObservationTermCfg(
        func=mdp.ball_predicted_edge_hit_point_b,
        params={
          "ball_cfg": _BALL_CFG,
          "robot_cfg": _ROBOT_CFG,
          "hit_height_offset": HIT_POINT_HEIGHT_OFFSET,
          "max_horizon": HIT_POINT_MAX_HORIZON,
        },
      ),
    }
  )

  observations = {
    "actor": ObservationGroupCfg(
      actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }
  actions: dict[str, ActionTermCfg] = {
    "latent_joint_pos": FrozenDecoderLatentJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
      latent_dim=16,
      decoder_state_terms=DECODER_STATE_TERMS,
    )
  }
  ball_provider_cfg = _ball_provider_cfg()
  events = {
    "reset_robot_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": ROBOT_RESET_X_RANGE,
          "y": ROBOT_RESET_Y_RANGE,
          "yaw": (ROBOT_RESET_YAW, ROBOT_RESET_YAW),
        },
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.02, 0.02),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": _ROBOT_CFG,
      },
    ),
    "reset_ball": EventTermCfg(
      func=mdp.spawn_ball_from_provider,
      mode="reset",
      params={"provider_cfg": ball_provider_cfg},
    ),
    "reset_table": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
        "asset_cfg": _TABLE_CFG,
      },
    ),
  }
  paddle_ball_sensor = ContactSensorCfg(
    name=_PADDLE_BALL_SENSOR,
    primary=ContactMatch(mode="geom", pattern="pingpong_ball", entity="ball"),
    secondary=ContactMatch(
      mode="geom",
      pattern="pingpong_paddle_collision",
      entity="robot",
    ),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  ball_net_sensor = ContactSensorCfg(
    name=_BALL_NET_SENSOR,
    primary=ContactMatch(mode="geom", pattern="pingpong_ball", entity="ball"),
    secondary=ContactMatch(
      mode="geom",
      pattern="pingpong_net_collision",
      entity="table",
    ),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  robot_table_sensor = ContactSensorCfg(
    name=_ROBOT_TABLE_SENSOR,
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(
      mode="geom",
      pattern=r"pingpong_(table_top_collision|net_collision)",
      entity="table",
    ),
    fields=("found", "force"),
    reduce="none",
    num_slots=8,
    history_length=4,
  )
  robot_ball_sensor = ContactSensorCfg(
    name=_ROBOT_BALL_SENSOR,
    primary=ContactMatch(
      mode="geom",
      pattern=r".*_collision",
      entity="robot",
      exclude=("pingpong_paddle_collision",),
    ),
    secondary=ContactMatch(mode="geom", pattern="pingpong_ball", entity="ball"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  state_params = _state_params()
  rewards = {
    "self_table_bounce_event": RewardTermCfg(
      func=mdp.self_table_bounce_event,
      weight=5.0,
      params=dict(state_params),
    ),
    "approach_ball": RewardTermCfg(
      func=mdp.paddle_to_ball_after_bounce_dense,
      weight=5.0,
      params={
        **dict(state_params),
        "std": 0.35,
        "paddle_cfg": _PADDLE_CFG,
        "robot_cfg": _ROBOT_CFG,
      },
    ),
    "paddle_towards_ball": RewardTermCfg(
      func=mdp.paddle_towards_ball_velocity,
      weight=2.0,
      params={
        **dict(state_params),
        "paddle_cfg": _PADDLE_CFG,
        "robot_cfg": _ROBOT_CFG,
        "speed_scale": 1.8,
        "distance_std": 0.55,
      },
    ),
    "paddle_hit_event": RewardTermCfg(
      func=mdp.paddle_hit_event,
      weight=2000.0,
      params=dict(state_params),
    ),
    "alive": RewardTermCfg(func=mdp.is_alive, weight=0.01),
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "joint_torques_l2": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-2e-5,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "joint_acc_l2": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-2e-6,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "latent_action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005),
    "low_level_action_rate_l2": RewardTermCfg(
      func=mdp.low_level_action_rate_l2,
      weight=-0.02,
      params={"action_name": "latent_joint_pos"},
    ),
    "robot_table_contact": RewardTermCfg(
      func=mdp.robot_table_contact_penalty,
      weight=-5.0,
      params={
        "sensor_name": _ROBOT_TABLE_SENSOR,
        "force_threshold": 5.0,
        "max_count": 4.0,
      },
    ),
    "robot_ball_contact": RewardTermCfg(
      func=mdp.robot_ball_contact_penalty,
      weight=-50.0,
      params={
        "sensor_name": _ROBOT_BALL_SENSOR,
        "force_threshold": HIT_FORCE_THRESHOLD,
        "max_count": 4.0,
      },
    ),
    "fall_penalty": RewardTermCfg(
      func=mdp.termination_terms_any,
      weight=-200.0,
      params={"term_names": ("bad_orientation", "root_height")},
    ),
    "flat_orientation_l2": RewardTermCfg(
      func=mdp.flat_orientation_l2,
      weight=0.0,
      params={"asset_cfg": _ROBOT_CFG},
    ),
  }
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "nan_detection": TerminationTermCfg(func=mdp.nan_detection),
    "bad_orientation": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "root_height": TerminationTermCfg(
      func=mdp.root_height_below_minimum,
      params={"minimum_height": 0.45},
    ),
    "ball_fault": TerminationTermCfg(
      func=mdp.pingpong_ball_fault,
      params=dict(state_params),
    ),
    "first_paddle_hit": TerminationTermCfg(
      func=mdp.first_paddle_hit,
      params=dict(state_params),
    ),
  }
  metrics = {
    "self_table_bounce_count": MetricsTermCfg(
      func=mdp.self_table_bounce_count_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "paddle_hit_count": MetricsTermCfg(
      func=mdp.paddle_hit_count_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "crossed_net_count": MetricsTermCfg(
      func=mdp.crossed_net_count_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "opponent_table_bounce_count": MetricsTermCfg(
      func=mdp.opponent_table_bounce_count_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "legal_return_count": MetricsTermCfg(
      func=mdp.legal_return_count_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "fault_count": MetricsTermCfg(
      func=mdp.fault_count_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "robot_table_contact_count": MetricsTermCfg(
      func=mdp.robot_table_contact_count_metric,
      reduce="last",
      params={
        "sensor_name": _ROBOT_TABLE_SENSOR,
        "force_threshold": 5.0,
      },
    ),
    "robot_ball_contact_count": MetricsTermCfg(
      func=mdp.robot_ball_contact_count_metric,
      reduce="last",
      params={
        "sensor_name": _ROBOT_BALL_SENSOR,
        "force_threshold": HIT_FORCE_THRESHOLD,
      },
    ),
    "fault_reason/body_ball": MetricsTermCfg(
      func=mdp.fault_reason_body_ball_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "fault_reason/low_net": MetricsTermCfg(
      func=mdp.fault_reason_low_net_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "fault_reason/net_contact": MetricsTermCfg(
      func=mdp.fault_reason_net_contact_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "fault_reason/return_out": MetricsTermCfg(
      func=mdp.fault_reason_return_out_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "fault_reason/failed_bounce": MetricsTermCfg(
      func=mdp.fault_reason_failed_bounce_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "fault_reason/double_paddle": MetricsTermCfg(
      func=mdp.fault_reason_double_paddle_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "fault_reason/early_hit": MetricsTermCfg(
      func=mdp.fault_reason_early_hit_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "hit/post_vx": MetricsTermCfg(
      func=mdp.hit_post_vx_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "hit/post_vy": MetricsTermCfg(
      func=mdp.hit_post_vy_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "hit/post_vz": MetricsTermCfg(
      func=mdp.hit_post_vz_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "hit/post_speed": MetricsTermCfg(
      func=mdp.hit_post_speed_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "hit/post_vx_toward_opponent_ratio": MetricsTermCfg(
      func=mdp.hit_post_vx_toward_opponent_ratio_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "hit/pred_net_clearance": MetricsTermCfg(
      func=mdp.hit_pred_net_clearance_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "hit/pred_net_clearance_positive": MetricsTermCfg(
      func=mdp.hit_pred_net_clearance_positive_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "hit/pred_landing_x": MetricsTermCfg(
      func=mdp.hit_pred_landing_x_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "hit/pred_landing_y": MetricsTermCfg(
      func=mdp.hit_pred_landing_y_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "hit/pred_landing_inside_opponent_table": MetricsTermCfg(
      func=mdp.hit_pred_landing_inside_opponent_table_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "hit/paddle_speed": MetricsTermCfg(
      func=mdp.hit_paddle_speed_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "hit/paddle_normal_alignment": MetricsTermCfg(
      func=mdp.hit_paddle_normal_alignment_metric,
      reduce="last",
      params=dict(state_params),
    ),
    "hit/paddle_velocity_along_normal": MetricsTermCfg(
      func=mdp.hit_paddle_velocity_along_normal_metric,
      reduce="last",
      params=dict(state_params),
    ),
  }
  curriculum = {
    "ball_target_region": CurriculumTermCfg(
      func=mdp.random_feeder_target_curriculum,
      params={
        "provider_cfg": ball_provider_cfg,
        "initial_target_x_range": BALL_TARGET_INITIAL_X_RANGE,
        "initial_target_y_range": BALL_TARGET_INITIAL_Y_RANGE,
        "final_target_x_range": BALL_TARGET_X_RANGE,
        "final_target_y_range": BALL_TARGET_Y_RANGE,
        "success_term_name": "first_paddle_hit",
        "success_threshold": BALL_TARGET_CURRICULUM_SUCCESS_THRESHOLD,
        "success_window": BALL_TARGET_CURRICULUM_WINDOW,
        "num_stages": BALL_TARGET_CURRICULUM_STAGES,
      },
    ),
    "action_regularization": CurriculumTermCfg(
      func=mdp.success_reward_weight_curriculum,
      params={
        "success_term_name": "first_paddle_hit",
        "success_threshold": ACTION_REGULARIZATION_CURRICULUM_SUCCESS_THRESHOLD,
        "success_window": ACTION_REGULARIZATION_CURRICULUM_WINDOW,
        "stage_weights": ACTION_REGULARIZATION_CURRICULUM_STAGE_WEIGHTS,
        "prerequisite_curriculum_name": "ball_target_region",
        "prerequisite_stage_key": "stage",
        "prerequisite_min_stage": float(BALL_TARGET_CURRICULUM_STAGES - 1),
      },
    ),
  }
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=get_pingpong_terrain_cfg(),
      entities={
        "ball": get_pingpong_ball_cfg(),
        "table": get_pingpong_table_cfg(),
      },
      sensors=(
        paddle_ball_sensor,
        ball_net_sensor,
        robot_table_sensor,
        robot_ball_sensor,
      ),
      num_envs=1,
      env_spacing=5.0,
      extent=3.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    metrics=metrics,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=3.0,
      elevation=-16.0,
      azimuth=135.0,
      fovy=55.0,
    ),
    sim=SimulationCfg(
      nconmax=96,
      njmax=900,
      contact_sensor_maxmatch=64,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=50,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=4,
    episode_length_s=3.0,
  )


def _add_strike_quality_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  state_params = _state_params()
  cfg.rewards["strike_pred_net_clearance"] = RewardTermCfg(
    func=mdp.strike_pred_net_clearance,
    weight=CROSS_STRIKE_QUALITY_REWARD_WEIGHTS["strike_pred_net_clearance"],
    params={
      **dict(state_params),
      "clearance_margin": 0.03,
      "clearance_scale": 0.18,
    },
  )
  cfg.rewards["strike_pred_landing_inside"] = RewardTermCfg(
    func=mdp.strike_pred_landing_inside,
    weight=CROSS_STRIKE_QUALITY_REWARD_WEIGHTS["strike_pred_landing_inside"],
    params=dict(state_params),
  )
  cfg.rewards["strike_post_hit_speed"] = RewardTermCfg(
    func=mdp.strike_post_hit_speed,
    weight=CROSS_STRIKE_QUALITY_REWARD_WEIGHTS["strike_post_hit_speed"],
    params={
      **dict(state_params),
      "speed_scale": 4.0,
    },
  )


def _add_impact_window_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  del cfg


def _add_impact_window_metrics(cfg: ManagerBasedRlEnvCfg) -> None:
  state_params = _state_params()
  cfg.metrics["impact/window_active"] = MetricsTermCfg(
    func=mdp.impact_window_active_metric,
    reduce="mean",
    params=dict(state_params),
  )
  cfg.metrics["impact/window_count"] = MetricsTermCfg(
    func=mdp.impact_window_count_metric,
    reduce="last",
    params=dict(state_params),
  )
  cfg.metrics["impact/paddle_speed"] = MetricsTermCfg(
    func=mdp.impact_paddle_speed_metric,
    reduce="last",
    params=dict(state_params),
  )
  cfg.metrics["impact/velocity_to_target"] = MetricsTermCfg(
    func=mdp.impact_velocity_to_target_metric,
    reduce="last",
    params=dict(state_params),
  )
  cfg.metrics["impact/velocity_along_normal"] = MetricsTermCfg(
    func=mdp.impact_velocity_along_normal_metric,
    reduce="last",
    params=dict(state_params),
  )
  cfg.metrics["impact/normal_to_target"] = MetricsTermCfg(
    func=mdp.impact_normal_to_target_metric,
    reduce="last",
    params=dict(state_params),
  )
  cfg.metrics["impact/center_distance"] = MetricsTermCfg(
    func=mdp.impact_center_distance_metric,
    reduce="last",
    params=dict(state_params),
  )
  cfg.metrics["impact/followthrough_velocity"] = MetricsTermCfg(
    func=mdp.impact_followthrough_velocity_metric,
    reduce="last",
    params=dict(state_params),
  )


def _apply_hit_window_energy_relaxation(cfg: ManagerBasedRlEnvCfg) -> None:
  state_params = _state_params()
  cfg.rewards["latent_action_rate_l2"] = RewardTermCfg(
    func=mdp.pre_hit_action_rate_l2,
    weight=CROSS_LOOSE_REGULARIZATION_WEIGHTS["latent_action_rate_l2"],
    params=dict(state_params),
  )
  cfg.rewards["low_level_action_rate_l2"] = RewardTermCfg(
    func=mdp.pre_hit_low_level_action_rate_l2,
    weight=CROSS_LOOSE_REGULARIZATION_WEIGHTS["low_level_action_rate_l2"],
    params={**dict(state_params), "action_name": "latent_joint_pos"},
  )


def make_pingpong_latent_cross_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the legal over-net table-tennis return task."""
  cfg = make_pingpong_latent_env_cfg()
  state_params = _state_params()
  cfg.rewards.pop("approach_ball", None)
  cfg.rewards.pop("paddle_towards_ball", None)
  cfg.rewards["hit_point"] = RewardTermCfg(
    func=mdp.paddle_to_predicted_hit_point_dense,
    weight=CROSS_HIT_POINT_WEIGHT,
    params={
      **dict(state_params),
      "std": 0.35,
      "paddle_cfg": _PADDLE_CFG,
      "robot_cfg": _ROBOT_CFG,
    },
  )
  cfg.rewards["paddle_hit_event"].weight = 200.0
  for reward_name, reward_weight in CROSS_LOOSE_REGULARIZATION_WEIGHTS.items():
    cfg.rewards[reward_name].weight = reward_weight
  cfg.rewards["robot_ball_contact"].weight = CROSS_ROBOT_BALL_CONTACT_WEIGHT
  cfg.rewards["post_hit_x_progress"] = RewardTermCfg(
    func=mdp.post_hit_x_progress,
    weight=CROSS_POST_HIT_X_PROGRESS_WEIGHT,
    params={
      **dict(state_params),
      "max_progress": 0.04,
      "lateral_speed_std": 0.8,
    },
  )
  cfg.rewards["post_hit_ball_velocity_direction"] = RewardTermCfg(
    func=mdp.post_hit_ball_velocity_direction,
    weight=CROSS_POST_HIT_BALL_VELOCITY_DIRECTION_WEIGHT,
    params={
      **dict(state_params),
      "x_speed_scale": 2.5,
      "lateral_speed_std": 0.8,
    },
  )
  cfg.rewards["crossed_net_event"] = RewardTermCfg(
    func=mdp.crossed_net_event,
    weight=500.0,
    params=dict(state_params),
  )
  cfg.rewards["opponent_table_bounce_event"] = RewardTermCfg(
    func=mdp.opponent_table_bounce_event,
    weight=CROSS_LAND_OPPONENT_WEIGHT,
    params=dict(state_params),
  )
  cfg.terminations.pop("first_paddle_hit", None)
  cfg.terminations["legal_return_success"] = TerminationTermCfg(
    func=mdp.legal_return_success,
    params=dict(state_params),
  )
  cfg.terminations["bad_orientation"].params["limit_angle"] = math.radians(55.0)
  cfg.terminations["root_height"].params["minimum_height"] = 0.55
  cfg.curriculum["ball_target_region"].params["success_term_name"] = (
    "legal_return_success"
  )
  cfg.curriculum.pop("action_regularization", None)
  return cfg


def make_pingpong_latent_cross_diag_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create a diagnostics-only Cross ablation without extra strike rewards."""
  return make_pingpong_latent_cross_env_cfg()


def make_pingpong_latent_cross_strike_quality_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create a Cross ablation with conservative strike-quality dense rewards."""
  cfg = make_pingpong_latent_cross_env_cfg()
  _add_strike_quality_rewards(cfg)
  return cfg


def make_pingpong_latent_cross_impact_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create a Cross ablation with impact-window paddle behavior rewards."""
  cfg = make_pingpong_latent_cross_strike_quality_env_cfg()
  _add_impact_window_rewards(cfg)
  _add_impact_window_metrics(cfg)
  return cfg


def make_pingpong_latent_cross_strike_quality_energy_relax_env_cfg() -> (
  ManagerBasedRlEnvCfg
):
  """Create a Cross ablation with strike rewards and hit-window energy relax."""
  cfg = make_pingpong_latent_cross_strike_quality_env_cfg()
  _apply_hit_window_energy_relaxation(cfg)
  return cfg


def make_pingpong_pace_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the PACE-style direct joint-control table-tennis return task."""
  cfg = make_pingpong_latent_cross_env_cfg()
  state_params = _pace_state_params()
  feet_cfg = SceneEntityCfg("robot", body_names=(".*ankle_roll_link",))
  left_foot_cfg = SceneEntityCfg("robot", body_names=("left_ankle_roll_link",))
  right_foot_cfg = SceneEntityCfg("robot", body_names=("right_ankle_roll_link",))
  right_shoulder_cfg = SceneEntityCfg("robot", body_names=("right_shoulder_pitch_link",))
  right_elbow_cfg = SceneEntityCfg("robot", body_names=("right_elbow_link",))
  right_wrist_cfg = SceneEntityCfg("robot", body_names=("right_wrist_yaw_link",))
  ankle_cfg = SceneEntityCfg(
    "robot",
    joint_names=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  )
  left_arm_cfg = SceneEntityCfg(
    "robot",
    joint_names=("left_shoulder_.*", "left_elbow_joint"),
  )
  right_arm_cfg = SceneEntityCfg(
    "robot",
    joint_names=("right_shoulder_.*", "right_elbow_joint"),
  )
  hip_cfg = SceneEntityCfg(
    "robot",
    joint_names=(".*_hip_yaw_joint", ".*_hip_roll_joint"),
  )
  waist_cfg = SceneEntityCfg("robot", joint_names=("waist_.*_joint",))

  cfg.observations = _pace_observation_terms()
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.18,
      use_default_offset=True,
    )
  }
  foot_contact_sensor = ContactSensorCfg(
    name=PACE_FOOT_CONTACT_SENSOR,
    primary=ContactMatch(
      mode="geom",
      pattern=PACE_FOOT_GEOM_NAMES,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
    history_length=4,
  )
  cfg.scene.sensors = (*cfg.scene.sensors, foot_contact_sensor)
  cfg.rewards = {
    "lin_vel_z_l2": RewardTermCfg(
      func=mdp.pace_lin_vel_z_l2,
      weight=-1.0,
      params={"robot_cfg": _ROBOT_CFG},
    ),
    "ang_vel_xy_l2": RewardTermCfg(
      func=mdp.pace_ang_vel_xy_l2,
      weight=-0.05,
      params={"robot_cfg": _ROBOT_CFG},
    ),
    "ang_vel_z_l2": RewardTermCfg(
      func=mdp.pace_ang_vel_z_l2,
      weight=-0.02,
      params={"robot_cfg": _ROBOT_CFG},
    ),
    "energy": RewardTermCfg(
      func=mdp.electrical_power_cost,
      weight=-1.5e-3,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "energy_ankle": RewardTermCfg(
      func=mdp.electrical_power_cost,
      weight=-2.0e-3,
      params={"asset_cfg": ankle_cfg},
    ),
    "joint_acc_l2": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-1.25e-7,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "robot_table_contact": RewardTermCfg(
      func=mdp.robot_table_contact_penalty,
      weight=-80.0,
      params={
        "sensor_name": _ROBOT_TABLE_SENSOR,
        "force_threshold": 1.0,
        "max_count": 4.0,
      },
    ),
    "robot_ball_contact": RewardTermCfg(
      func=mdp.robot_ball_contact_penalty,
      weight=-80.0,
      params={
        "sensor_name": _ROBOT_BALL_SENSOR,
        "force_threshold": HIT_FORCE_THRESHOLD,
        "max_count": 4.0,
      },
    ),
    "robot_table_proximity_x": RewardTermCfg(
      func=mdp.pace_robot_table_proximity_x,
      weight=-20.0,
      params={"robot_cfg": _ROBOT_CFG, "min_distance": 0.15},
    ),
    "flat_orientation_l2": RewardTermCfg(
      func=mdp.flat_orientation_l2,
      weight=-1.5,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "termination_penalty": RewardTermCfg(func=mdp.is_terminated, weight=-1000.0),
    "pace_fly": RewardTermCfg(
      func=mdp.pace_fly,
      weight=-2.5,
      params={"sensor_name": PACE_FOOT_CONTACT_SENSOR, "force_threshold": 1.0},
    ),
    "pace_hit_unstable_support": RewardTermCfg(
      func=mdp.pace_hit_unstable_support,
      weight=-5.0,
      params={
        **dict(state_params),
        "sensor_name": PACE_FOOT_CONTACT_SENSOR,
        "force_threshold": 0.1,
      },
    ),
    "pace_feet_orientation_left": RewardTermCfg(
      func=mdp.pace_body_orientation_l2,
      weight=-4.0,
      params={"body_cfg": left_foot_cfg},
    ),
    "pace_feet_orientation_right": RewardTermCfg(
      func=mdp.pace_body_orientation_l2,
      weight=-4.0,
      params={"body_cfg": right_foot_cfg},
    ),
    "feet_slide": RewardTermCfg(
      func=mdp.pace_feet_slide_contact,
      weight=-0.3,
      params={
        "feet_cfg": feet_cfg,
        "sensor_name": PACE_FOOT_CONTACT_SENSOR,
        "force_threshold": 1.0,
      },
    ),
    "feet_force": RewardTermCfg(
      func=mdp.pace_feet_force,
      weight=-3.0e-3,
      params={
        "sensor_name": PACE_FOOT_CONTACT_SENSOR,
        "threshold": 500.0,
        "max_reward": 400.0,
      },
    ),
    "feet_stumble": RewardTermCfg(
      func=mdp.pace_feet_stumble,
      weight=-2.0,
      params={"sensor_name": PACE_FOOT_CONTACT_SENSOR},
    ),
    "pace_feet_too_near": RewardTermCfg(
      func=mdp.pace_feet_too_near,
      weight=-1.5,
      params={"feet_cfg": feet_cfg, "threshold": 0.20},
    ),
    "pace_feet_really_too_near": RewardTermCfg(
      func=mdp.pace_feet_too_near,
      weight=-10.0,
      params={"feet_cfg": feet_cfg, "threshold": 0.15},
    ),
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-2.0,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "joint_deviation_hip": RewardTermCfg(
      func=mdp.pace_joint_deviation_l1,
      weight=-0.2,
      params={"asset_cfg": hip_cfg},
    ),
    "joint_deviation_left_arm": RewardTermCfg(
      func=mdp.pace_joint_deviation_l1,
      weight=-0.2,
      params={"asset_cfg": left_arm_cfg},
    ),
    "joint_deviation_right_arm": RewardTermCfg(
      func=mdp.pace_joint_deviation_l1,
      weight=-0.05,
      params={"asset_cfg": right_arm_cfg},
    ),
    "joint_deviation_torso": RewardTermCfg(
      func=mdp.pace_joint_deviation_l1,
      weight=-0.2,
      params={"asset_cfg": waist_cfg},
    ),
    "pace_contact": RewardTermCfg(
      func=mdp.pace_contact,
      weight=PACE_TASK_REWARD_WEIGHTS["pace_contact"],
      params=dict(state_params),
    ),
    "pace_future_ee_target": RewardTermCfg(
      func=mdp.pace_future_ee_target,
      weight=PACE_TASK_REWARD_WEIGHTS["pace_future_ee_target"],
      params={**dict(state_params), "std_ee": 0.5, "threshold": 0.15},
    ),
    "pace_future_paddle_height_target": RewardTermCfg(
      func=mdp.pace_future_paddle_height_target,
      weight=PACE_TASK_REWARD_WEIGHTS["pace_future_paddle_height_target"],
      params={**dict(state_params), "z_std": 0.25},
    ),
    "pace_future_body_target": RewardTermCfg(
      func=mdp.pace_future_body_target,
      weight=PACE_TASK_REWARD_WEIGHTS["pace_future_body_target"],
      params={**dict(state_params), "std_ro": 0.5, "threshold": 0.05},
    ),
    "pace_future_base_vel_target": RewardTermCfg(
      func=mdp.pace_future_base_vel_target,
      weight=PACE_TASK_REWARD_WEIGHTS["pace_future_base_vel_target"],
      params={**dict(state_params), "vel_std": 1.2, "threshold": 0.1},
    ),
    "pace_future_landing_distance": RewardTermCfg(
      func=mdp.pace_future_landing_distance,
      weight=PACE_TASK_REWARD_WEIGHTS["pace_future_landing_distance"],
      params={**dict(state_params), "threshold": 3.0},
    ),
    "pace_future_pass_net": RewardTermCfg(
      func=mdp.pace_future_pass_net,
      weight=PACE_TASK_REWARD_WEIGHTS["pace_future_pass_net"],
      params={**dict(state_params), "std_h": 0.4, "z_target": NET_TOP_Z + 0.35},
    ),
    "pace_table_success": RewardTermCfg(
      func=mdp.pace_table_success,
      weight=PACE_TASK_REWARD_WEIGHTS["pace_table_success"],
      params=dict(state_params),
    ),
    "pace_forehand_paddle_offset": RewardTermCfg(
      func=mdp.pace_forehand_paddle_offset,
      weight=PACE_TASK_REWARD_WEIGHTS["pace_forehand_paddle_offset"],
      params={
        **dict(state_params),
        "paddle_cfg": _PADDLE_CFG,
        "robot_cfg": _ROBOT_CFG,
        "target_offset": PACE_FOREHAND_PADDLE_OFFSET,
        "offset_std": PACE_FOREHAND_PADDLE_OFFSET_STD,
      },
    ),
    "pace_forehand_elbow_extension": RewardTermCfg(
      func=mdp.pace_forehand_elbow_extension,
      weight=PACE_TASK_REWARD_WEIGHTS["pace_forehand_elbow_extension"],
      params={
        **dict(state_params),
        "shoulder_cfg": right_shoulder_cfg,
        "elbow_cfg": right_elbow_cfg,
        "wrist_cfg": right_wrist_cfg,
        "target_ratio": PACE_FOREHAND_ELBOW_TARGET_RATIO,
        "std": 0.08,
      },
    ),
    "pace_step_air_time": RewardTermCfg(
      func=mdp.pace_step_air_time,
      weight=PACE_TASK_REWARD_WEIGHTS["pace_step_air_time"],
      params={
        **dict(state_params),
        "sensor_name": PACE_FOOT_CONTACT_SENSOR,
        "threshold_min": 0.05,
        "threshold_max": 0.50,
        "future_time_threshold": 0.18,
        "target_speed_threshold": 0.50,
      },
    ),
  }
  cfg.metrics["pace/target_valid_rate"] = MetricsTermCfg(
    func=mdp.pace_target_valid_metric,
    reduce="mean",
    params=dict(state_params),
  )
  cfg.metrics["pace/post_bounce_direct_prediction_rate"] = MetricsTermCfg(
    func=mdp.pace_prediction_post_bounce_direct_metric,
    reduce="mean",
    params=dict(state_params),
  )
  cfg.metrics["pace/posture_gate_mean"] = MetricsTermCfg(
    func=mdp.pace_posture_gate_metric,
    reduce="mean",
    params=dict(state_params),
  )
  cfg.metrics["pace/active_future_z_mean"] = MetricsTermCfg(
    func=mdp.pace_active_future_z_metric,
    reduce="mean",
    params=dict(state_params),
  )
  cfg.metrics["pace/active_paddle_z_mean"] = MetricsTermCfg(
    func=mdp.pace_active_paddle_z_metric,
    reduce="mean",
    params=dict(state_params),
  )
  cfg.metrics["pace/active_ee_dist_mean"] = MetricsTermCfg(
    func=mdp.pace_active_ee_dist_metric,
    reduce="mean",
    params=dict(state_params),
  )
  cfg.metrics["pace/active_target_base_speed_mean"] = MetricsTermCfg(
    func=mdp.pace_active_target_base_speed_metric,
    reduce="mean",
    params=dict(state_params),
  )
  cfg.metrics["pace/active_root_speed_mean"] = MetricsTermCfg(
    func=mdp.pace_active_root_speed_metric,
    reduce="mean",
    params=dict(state_params),
  )
  cfg.metrics["pace/invalid_not_moving"] = MetricsTermCfg(
    func=mdp.pace_target_invalid_not_moving_metric,
    reduce="mean",
    params=dict(state_params),
  )
  cfg.metrics["pace/invalid_bad_bounce"] = MetricsTermCfg(
    func=mdp.pace_target_invalid_bad_bounce_metric,
    reduce="mean",
    params=dict(state_params),
  )
  cfg.metrics["pace/invalid_second_bounce"] = MetricsTermCfg(
    func=mdp.pace_target_invalid_second_bounce_metric,
    reduce="mean",
    params=dict(state_params),
  )
  cfg.metrics["pace/invalid_out_of_bounds"] = MetricsTermCfg(
    func=mdp.pace_target_invalid_out_of_bounds_metric,
    reduce="mean",
    params=dict(state_params),
  )
  cfg.metrics["pace/invalid_low_or_time"] = MetricsTermCfg(
    func=mdp.pace_target_invalid_low_or_time_metric,
    reduce="mean",
    params=dict(state_params),
  )
  cfg.metrics["pace/invalid_numeric"] = MetricsTermCfg(
    func=mdp.pace_target_invalid_numeric_metric,
    reduce="mean",
    params=dict(state_params),
  )
  cfg.metrics["pace/invalid_rally_done"] = MetricsTermCfg(
    func=mdp.pace_target_invalid_rally_done_metric,
    reduce="mean",
    params=dict(state_params),
  )
  cfg.terminations["bad_orientation"].params["limit_angle"] = (
    PACE_BAD_ORIENTATION_LIMIT
  )
  cfg.terminations["root_height"].params["minimum_height"] = (
    PACE_ROOT_HEIGHT_MINIMUM
  )
  cfg.curriculum.pop("action_regularization", None)
  cfg.curriculum["ball_target_region"].params["success_term_name"] = (
    "legal_return_success"
  )
  return cfg


def make_pingpong_latent_return_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the legacy legal-return alias for the pingpong Cross task."""
  return make_pingpong_latent_cross_env_cfg()


__all__ = [
  "BALL_TARGET_X_RANGE",
  "BALL_TARGET_Y_RANGE",
  "CROSS_LOOSE_REGULARIZATION_WEIGHTS",
  "CROSS_HIT_POINT_WEIGHT",
  "CROSS_LAND_OPPONENT_WEIGHT",
  "CROSS_POST_HIT_BALL_VELOCITY_DIRECTION_WEIGHT",
  "CROSS_POST_HIT_X_PROGRESS_WEIGHT",
  "CROSS_ROBOT_BALL_CONTACT_WEIGHT",
  "CROSS_IMPACT_REWARD_WEIGHTS",
  "CROSS_STRIKE_QUALITY_REWARD_WEIGHTS",
  "DECODER_STATE_TERMS",
  "PACE_TASK_REWARD_WEIGHTS",
  "PACE_BAD_ORIENTATION_LIMIT",
  "PACE_FOOT_CONTACT_SENSOR",
  "PACE_FOOT_GEOM_NAMES",
  "PACE_FOREHAND_ELBOW_TARGET_RATIO",
  "PACE_FOREHAND_PADDLE_OFFSET",
  "PACE_FOREHAND_PADDLE_OFFSET_STD",
  "PACE_NATURAL_HIT_X",
  "PACE_ROOT_HEIGHT_MINIMUM",
  "PACE_TARGET_BASE_OFFSET_XY",
  "PACE_TARGET_BASE_VEL_GAIN",
  "PACE_TARGET_BASE_VEL_MAX",
  "PACE_TARGET_ROOT_HEIGHT",
  "ACTION_REGULARIZATION_CURRICULUM_STAGE_WEIGHTS",
  "make_pingpong_latent_cross_diag_env_cfg",
  "make_pingpong_latent_cross_env_cfg",
  "make_pingpong_latent_cross_impact_env_cfg",
  "make_pingpong_latent_env_cfg",
  "make_pingpong_latent_return_env_cfg",
  "make_pingpong_latent_cross_strike_quality_energy_relax_env_cfg",
  "make_pingpong_latent_cross_strike_quality_env_cfg",
  "make_pingpong_pace_env_cfg",
]
