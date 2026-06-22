"""Table-tennis latent-control task configuration."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
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
from mjlab.tasks.pingpong.mdp.ball_providers import TableTennisFeederCfg
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
_BALL_CFG = SceneEntityCfg("ball")
_TABLE_CFG = SceneEntityCfg("table")
_PADDLE_BALL_SENSOR = "paddle_ball_contact"
_BALL_NET_SENSOR = "pingpong_ball_net_contact"

ROBOT_RESET_X_RANGE = (TABLE_HALF_LENGTH + 0.36, TABLE_HALF_LENGTH + 0.58)
ROBOT_RESET_Y_RANGE = (-0.18, 0.18)
ROBOT_RESET_YAW = math.pi

BALL_TARGET_INITIAL_X_RANGE = (0.58, 0.72)
BALL_TARGET_INITIAL_Y_RANGE = (-0.10, 0.10)
BALL_TARGET_X_RANGE = (0.25, TABLE_HALF_LENGTH - 0.18)
BALL_TARGET_Y_RANGE = (-TABLE_HALF_WIDTH * 0.70, TABLE_HALF_WIDTH * 0.70)
BALL_TARGET_CURRICULUM_SUCCESS_THRESHOLD = 0.75
BALL_TARGET_CURRICULUM_WINDOW = 50
BALL_TARGET_CURRICULUM_STAGES = 6

OUT_X_LIMITS = (-TABLE_HALF_LENGTH - 0.75, TABLE_HALF_LENGTH + 1.10)
OUT_Y_LIMITS = (-TABLE_HALF_WIDTH - 0.50, TABLE_HALF_WIDTH + 0.50)
OUT_Z_LIMITS = (0.05, 2.5)
SELF_X_LIMITS = (NET_X, TABLE_HALF_LENGTH)
OPPONENT_X_LIMITS = (-TABLE_HALF_LENGTH, NET_X)
TABLE_Y_LIMITS = (-TABLE_HALF_WIDTH, TABLE_HALF_WIDTH)
HIT_FORCE_THRESHOLD = 0.05
BOUNCE_Z_TOLERANCE = 0.055

HIT_POINT_HEIGHT_OFFSET = 0.02
HIT_POINT_MAX_HORIZON = 1.2
DECODER_STATE_TERMS = (
  "base_lin_vel",
  "base_ang_vel",
  "joint_pos",
  "joint_vel",
  "actions",
)


def _state_params() -> dict[str, object]:
  return {
    "paddle_sensor_name": _PADDLE_BALL_SENSOR,
    "net_sensor_name": _BALL_NET_SENSOR,
    "ball_cfg": _BALL_CFG,
    "force_threshold": HIT_FORCE_THRESHOLD,
    "table_z": BALL_CENTER_TABLE_Z,
    "net_x": NET_X,
    "net_top_z": NET_TOP_Z,
    "self_x_limits": SELF_X_LIMITS,
    "opponent_x_limits": OPPONENT_X_LIMITS,
    "table_y_limits": TABLE_Y_LIMITS,
    "x_limits": OUT_X_LIMITS,
    "y_limits": OUT_Y_LIMITS,
    "z_limits": OUT_Z_LIMITS,
    "bounce_z_tolerance": BOUNCE_Z_TOLERANCE,
  }


def _ball_provider_cfg() -> TableTennisFeederCfg:
  return TableTennisFeederCfg(
    ball_cfg=_BALL_CFG,
    spawn_x_range=(-1.18, -0.28),
    spawn_y_range=(-TABLE_HALF_WIDTH * 0.55, TABLE_HALF_WIDTH * 0.55),
    spawn_z_range=(1.05, 1.35),
    target_x_range=BALL_TARGET_INITIAL_X_RANGE,
    target_y_range=BALL_TARGET_INITIAL_Y_RANGE,
    flight_time_range=(0.55, 0.85),
    target_z=BALL_CENTER_TABLE_Z,
    net_top_z=NET_TOP_Z,
    net_clearance=0.06,
  )


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
    func=mdp.racket_to_ball_b,
    params={
      "racket_cfg": _PADDLE_CFG,
      "ball_cfg": _BALL_CFG,
      "robot_cfg": _ROBOT_CFG,
    },
    noise=Unoise(n_min=-0.01, n_max=0.01),
    history_length=10,
    flatten_history_dim=True,
  )
  actor_terms["predicted_hit_point"] = ObservationTermCfg(
    func=mdp.ball_predicted_hit_point_b,
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
        func=mdp.ball_predicted_hit_point_b,
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
  state_params = _state_params()
  rewards = {
    "self_table_bounce_event": RewardTermCfg(
      func=mdp.self_table_bounce_event,
      weight=5.0,
      params=dict(state_params),
    ),
    "approach_ball": RewardTermCfg(
      func=mdp.paddle_to_ball_after_bounce_dense,
      weight=10.0,
      params={
        **dict(state_params),
        "std": 0.35,
        "paddle_cfg": _PADDLE_CFG,
        "robot_cfg": _ROBOT_CFG,
      },
    ),
    "paddle_towards_ball": RewardTermCfg(
      func=mdp.paddle_towards_ball_velocity,
      weight=5.0,
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
      weight=100.0,
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
    "fall_penalty": RewardTermCfg(
      func=mdp.termination_terms_any,
      weight=-200.0,
      params={"term_names": ("bad_orientation", "root_height")},
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
    )
  }
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=get_pingpong_terrain_cfg(),
      entities={
        "ball": get_pingpong_ball_cfg(),
        "table": get_pingpong_table_cfg(),
      },
      sensors=(paddle_ball_sensor, ball_net_sensor),
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


def make_pingpong_latent_return_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the legal single-return table-tennis task."""
  cfg = make_pingpong_latent_env_cfg()
  state_params = _state_params()
  cfg.rewards["approach_ball"].weight = 5.0
  cfg.rewards["paddle_towards_ball"].weight = 2.0
  cfg.rewards["paddle_hit_event"].weight = 25.0
  cfg.rewards["post_hit_x_progress"] = RewardTermCfg(
    func=mdp.post_hit_x_progress,
    weight=40.0,
    params={**dict(state_params), "max_progress": 0.04},
  )
  cfg.rewards["post_hit_ball_velocity_direction"] = RewardTermCfg(
    func=mdp.post_hit_ball_velocity_direction,
    weight=20.0,
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
    weight=1000.0,
    params=dict(state_params),
  )
  cfg.terminations.pop("first_paddle_hit", None)
  cfg.terminations["legal_return_success"] = TerminationTermCfg(
    func=mdp.legal_return_success,
    params=dict(state_params),
  )
  cfg.curriculum["ball_target_region"].params["success_term_name"] = (
    "legal_return_success"
  )
  return cfg


__all__ = [
  "BALL_TARGET_X_RANGE",
  "BALL_TARGET_Y_RANGE",
  "DECODER_STATE_TERMS",
  "make_pingpong_latent_env_cfg",
  "make_pingpong_latent_return_env_cfg",
]
