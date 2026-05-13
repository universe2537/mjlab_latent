"""Robot-agnostic tennis *return* task configuration.

This task is built around the new :class:`RallyCommand` finite-state machine
and a swappable ball provider. The default provider is the P1
:class:`RandomFeederCfg`, which mirrors the legacy hit-task ball reset and
keeps training behaviour close to ``Mjlab-Tennis-Hit-Unitree-G1`` while
exercising the new rule architecture.

Compared to ``tennis_env_cfg.make_tennis_latent_env_cfg``:

- ``reset_ball`` is removed; the rally command's ball provider spawns the
  ball on every reset.
- A new ``rally`` command term drives the FSM and event detection.
- The ``actor`` observation group includes the rally command vector.
- Rewards/terminations are wired to the rally state instead of the legacy
  ``TennisHitState``.

P0/P2 providers (``FixedSpawnerCfg``, ``BallisticOpponentCfg``) are
implemented in :mod:`mjlab.tasks.tennis.mdp.ball_providers` and can be
substituted by overriding ``cfg.commands["rally"].ball_provider`` in a
robot-specific factory.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tennis import mdp
from mjlab.tasks.tennis.mdp import FrozenDecoderLatentJointPositionActionCfg
from mjlab.tasks.tennis.mdp.ball_providers import (
  BallProviderCfg,
  RandomFeederCfg,
)
from mjlab.tasks.tennis.mdp.commands import RallyCommandCfg, RulesCfg
from mjlab.tasks.tennis.mdp.events import CourtBounds
from mjlab.tasks.tennis.scene import (
  get_tennis_ball_cfg,
  get_tennis_court_cfg,
  get_tennis_terrain_cfg,
)
from mjlab.tasks.tennis.tennis_env_cfg import (
  DECODER_STATE_TERMS,
  HIT_FORCE_THRESHOLD,
  ROBOT_RESET_X_RANGE,
  ROBOT_RESET_Y_RANGE,
  ROBOT_RESET_YAW,
  SUCCESS_TARGET_LINE_X,
  VALID_HIT_MIN_BALL_SPEED,
  VALID_HIT_MIN_LEFTWARD_SPEED,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

_ROBOT_CFG = SceneEntityCfg("robot", joint_names=(".*",))
_RACKET_CFG = SceneEntityCfg("robot", site_names=("tennis_racket_center",))
_BALL_CFG = SceneEntityCfg("ball")
_RACKET_BALL_SENSOR = "racket_ball_contact"
_RALLY_NAME = "rally"


def _default_ball_provider() -> BallProviderCfg:
  return RandomFeederCfg(
    ball_cfg=_BALL_CFG,
    # Spawn above the net, fly toward the robot's side.
    spawn_x_range=(-0.3, 0.3),
    spawn_y_range=(-2.0, 2.0),
    spawn_z_range=(0.9, 1.4),
    # Target landing zone: robot's side of the court.
    target_x_range=(0.5, 2.5),
    target_y_range=(-2.0, 2.0),
    # Vertical launch speed: positive = upward arc.
    lin_vel_z_range=(1.0, 3.0),
  )


def make_tennis_return_env_cfg(
  ball_provider: BallProviderCfg | None = None,
) -> ManagerBasedRlEnvCfg:
  """Build the rally-driven tennis return task base config.

  Parameters
  ----------
  ball_provider :
    Override the default ball provider (P1 RandomFeeder). Pass a
    :class:`FixedSpawnerCfg` for a deterministic eval setup or a
    :class:`BallisticOpponentCfg` for a more realistic incoming arc.
  """
  if ball_provider is None:
    ball_provider = _default_ball_provider()

  rally_cfg = RallyCommandCfg(
    ball_provider=ball_provider,
    rules=RulesCfg(
      bounds=CourtBounds(),
      hit_force_threshold=HIT_FORCE_THRESHOLD,
      valid_hit_min_leftward_speed=VALID_HIT_MIN_LEFTWARD_SPEED,
      valid_hit_min_ball_speed=VALID_HIT_MIN_BALL_SPEED,
      target_line_x=SUCCESS_TARGET_LINE_X,
    ),
    ball_cfg=_BALL_CFG,
    racket_ball_sensor=_RACKET_BALL_SENSOR,
    ball_net_sensor=None,
  )
  commands = {_RALLY_NAME: rally_cfg}

  # ---- Observations ---------------------------------------------------
  actor_terms = {
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
    "racket_to_ball": ObservationTermCfg(
      func=mdp.racket_to_ball_b,
      params={"racket_cfg": _RACKET_CFG, "ball_cfg": _BALL_CFG},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "ball_velocity": ObservationTermCfg(
      func=mdp.ball_velocity_b,
      params={"ball_cfg": _BALL_CFG},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "racket_velocity": ObservationTermCfg(
      func=mdp.racket_velocity_b,
      params={"racket_cfg": _RACKET_CFG},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "ball_landing": ObservationTermCfg(
      func=mdp.ball_predicted_landing_b,
      params={"ball_cfg": _BALL_CFG, "robot_cfg": _ROBOT_CFG},
    ),
    "rally_command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": _RALLY_NAME},
    ),
  }
  critic_terms = dict(actor_terms)
  observations = {
    "actor": ObservationGroupCfg(
      actor_terms, concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      critic_terms, concatenate_terms=True, enable_corruption=False
    ),
  }

  # ---- Actions --------------------------------------------------------
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

  # ---- Events: only the robot. The provider handles ball spawning. ----
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
    "reset_court": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("court"),
      },
    ),
  }

  # ---- Sensors --------------------------------------------------------
  hit_sensor = ContactSensorCfg(
    name=_RACKET_BALL_SENSOR,
    primary=ContactMatch(mode="geom", pattern="tennis_ball", entity="ball"),
    secondary=ContactMatch(
      mode="geom", pattern="tennis_racket_collision", entity="robot"
    ),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )

  # ---- Rewards --------------------------------------------------------
  rewards = {
    "alive": RewardTermCfg(func=mdp.is_alive, weight=0.01),
    # Bug fix ④: gated to pre-hit phases only; plain racket_ball_distance_exp
    # has no phase mask and keeps rewarding after the ball is struck.
    "approach_ball": RewardTermCfg(
      func=mdp.rally_approach_ball_pre_hit,
      weight=4.0,
      params={
        "command_name": _RALLY_NAME,
        "std": 0.4,
        "racket_cfg": _RACKET_CFG,
        "ball_cfg": _BALL_CFG,
        "robot_cfg": _ROBOT_CFG,
      },
    ),
    "valid_hit_event": RewardTermCfg(
      func=mdp.rally_valid_hit_event,
      weight=80.0,
      params={"command_name": _RALLY_NAME},
    ),
    "over_net": RewardTermCfg(
      func=mdp.rally_over_net_event,
      weight=20.0,
      params={"command_name": _RALLY_NAME},
    ),
    "point_won": RewardTermCfg(
      func=mdp.rally_point_won,
      weight=200.0,
      params={"command_name": _RALLY_NAME},
    ),
    "point_lost": RewardTermCfg(
      func=mdp.rally_point_lost,
      weight=-100.0,
      params={"command_name": _RALLY_NAME},
    ),
    "fall_penalty": RewardTermCfg(
      func=mdp.termination_terms_any,
      weight=-200.0,
      params={"term_names": ("bad_orientation", "root_height")},
    ),
    "posture": RewardTermCfg(
      func=mdp.posture,
      weight=0.2,
      params={
        "asset_cfg": _ROBOT_CFG,
        "std": {
          r".*hip.*": 0.25,
          r".*knee.*": 0.35,
          r".*ankle.*": 0.25,
          r".*waist.*": 0.25,
          r".*shoulder.*": 0.7,
          r".*elbow.*": 0.7,
          r".*wrist.*": 0.7,
        },
      },
    ),
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": _ROBOT_CFG}
    ),
    "joint_torques_l2": RewardTermCfg(
      func=mdp.joint_torques_l2, weight=-2e-5, params={"asset_cfg": _ROBOT_CFG}
    ),
    "joint_acc_l2": RewardTermCfg(
      func=mdp.joint_acc_l2, weight=-2e-6, params={"asset_cfg": _ROBOT_CFG}
    ),
    "latent_action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005),
    "low_level_action_rate_l2": RewardTermCfg(
      func=mdp.low_level_action_rate_l2,
      weight=-0.02,
      params={"action_name": "latent_joint_pos"},
    ),
  }

  # ---- Terminations ---------------------------------------------------
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "nan_detection": TerminationTermCfg(func=mdp.nan_detection),
    "bad_orientation": TerminationTermCfg(
      func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
    ),
    "root_height": TerminationTermCfg(
      func=mdp.root_height_below_minimum, params={"minimum_height": 0.45}
    ),
    "point_ended": TerminationTermCfg(
      func=mdp.point_ended, params={"command_name": _RALLY_NAME}
    ),
    "ball_out_of_bounds": TerminationTermCfg(
      func=mdp.ball_in_play,
      params={
        "x_limits": (-5.8, 3.6),
        "y_limits": (-2.7, 2.7),
        "z_limits": (0.05, 2.6),
      },
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=get_tennis_terrain_cfg(),
      entities={
        "ball": get_tennis_ball_cfg(),
        "court": get_tennis_court_cfg(),
      },
      sensors=(hit_sensor,),
      num_envs=1,
      env_spacing=4.0,
      extent=5.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    commands=commands,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=5.0,
      elevation=-18.0,
      azimuth=140.0,
      fovy=55.0,
    ),
    sim=SimulationCfg(
      nconmax=80,
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
    episode_length_s=4.0,
  )
  return cfg
