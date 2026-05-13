"""Robot-agnostic tennis Hit task configuration (refactored).

Refactor highlights
-------------------
* **Court** : G1-scaled tennis court built in :mod:`mjlab.tasks.tennis.scene`
              (14 m x 4.8 m total, net center 0.52 m).  See ``scene.py``.
* **Ball generation** : :class:`RandomFeederCfg` ballistic-trajectory feeder.
              Ball spawns above the net, target landing point is sampled on
              the robot's side, vz is sampled, and (vx, vy) are computed
              analytically from the kinematic inverse so the ball arcs into
              the target.
* **Terminations** :
              - ball out of court bounds
              - ball's second contact (racket OR ground)
              - ball successfully crosses the net after a hit
              - robot bad orientation / fell over
* **Rewards** : approach (1x), racket_hit_event (10x), crossed_net_event (100x),
              plus standard penalties (joint limits, torques, action rate, etc.).
* **Observations** :
              - **actor**: noisy proprioception + ball position window of 10
              - **critic**: full clean state (proprioception + ball/racket
                pos+vel + relative vector + predicted landing)
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
from mjlab.tasks.tennis.mdp.ball_providers import RandomFeederCfg
from mjlab.tasks.tennis.scene import (
  BASELINE_SELF_X,
  COURT_HALF_LENGTH,
  COURT_HALF_WIDTH,
  NET_CENTER_HEIGHT,
  get_tennis_ball_cfg,
  get_tennis_court_cfg,
  get_tennis_terrain_cfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

# ---------------------------------------------------------------------------
# Scene-entity configs (resolved at runtime by the scene).
# ---------------------------------------------------------------------------
_ROBOT_CFG = SceneEntityCfg("robot", joint_names=(".*",))
_RACKET_CFG = SceneEntityCfg("robot", site_names=("tennis_racket_center",))
_BALL_CFG = SceneEntityCfg("ball")
_COURT_CFG = SceneEntityCfg("court")
_RACKET_BALL_SENSOR = "racket_ball_contact"

# ---------------------------------------------------------------------------
# Robot reset (mid-court self side, facing the net).
# ---------------------------------------------------------------------------
ROBOT_RESET_X_RANGE = (3.5, 4.5)
ROBOT_RESET_Y_RANGE = (-0.4, 0.4)
ROBOT_RESET_YAW = math.pi  # face -x (toward the net / opponent)

# ---------------------------------------------------------------------------
# Tracker thresholds.
# ---------------------------------------------------------------------------
HIT_FORCE_THRESHOLD = 1.0
GROUND_Z = 0.06  # ball-radius (~0.034) + small margin
NET_X = 0.0

# ---------------------------------------------------------------------------
# Court bounds for "ball out of play" (slightly looser than painted lines).
# ---------------------------------------------------------------------------
COURT_OUT_X_LIMITS = (-COURT_HALF_LENGTH - 1.0, BASELINE_SELF_X + 1.0)
COURT_OUT_Y_LIMITS = (-COURT_HALF_WIDTH - 0.5, COURT_HALF_WIDTH + 0.5)
COURT_OUT_Z_LIMITS = (0.02, 3.0)

# ---------------------------------------------------------------------------
# Frozen low-level decoder state terms (must match distillation checkpoint).
# ---------------------------------------------------------------------------
DECODER_STATE_TERMS = (
  "base_lin_vel",
  "base_ang_vel",
  "joint_pos",
  "joint_vel",
  "actions",
)

# ---------------------------------------------------------------------------
# Legacy constants kept for backward compatibility with the Rally / Return
# task (return_env_cfg.py).  They are *not* used by the refactored Hit task,
# whose terminations are based on the simpler TennisRallyTracker (second
# contact / over-net) instead of the speed-thresholded "valid hit" notion.
# ---------------------------------------------------------------------------
VALID_HIT_MIN_LEFTWARD_SPEED = 2.0
VALID_HIT_MIN_BALL_SPEED = 2.5
SUCCESS_TARGET_LINE_X = -3.5  # rescaled from -2.2 for the larger court
MISS_BALL_X_OFFSET = 0.2
MISS_BALL_X_DIRECTION = 1.0


def make_tennis_latent_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the refactored Hit task with a frozen low-level latent decoder.

  Robot-specific modules fill in the robot asset, action scale, and viewer
  body. The actor sees decoder-compatible proprioception (with noise) plus
  a 10-step window of the ball position; the critic sees the same
  proprioception (clean) augmented with ball/racket velocities and the
  predicted ball landing point.
  """
  # -------------------------------------------------------------------------
  # Decoder-compatible proprioception (shared between actor / critic).
  # The action term will slice exactly DECODER_STATE_TERMS from the actor obs.
  # -------------------------------------------------------------------------
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

  # ---- Actor: noisy proprioception + ball position with 10-step window ---
  actor_terms = dict(proprio_actor)
  # Ball position relative to robot base, last 10 frames (flattened => 30 dims).
  actor_terms["ball_pos_window"] = ObservationTermCfg(
    func=mdp.racket_to_ball_b,
    params={
      "racket_cfg": _RACKET_CFG,
      "ball_cfg": _BALL_CFG,
      "robot_cfg": _ROBOT_CFG,
    },
    noise=Unoise(n_min=-0.01, n_max=0.01),
    history_length=10,
    flatten_history_dim=True,
  )

  # ---- Critic: clean proprioception + as much state as possible ----------
  critic_terms = {
    name: ObservationTermCfg(func=t.func, params=dict(t.params))
    for name, t in proprio_actor.items()
  }
  critic_terms.update(
    {
      "racket_to_ball": ObservationTermCfg(
        func=mdp.racket_to_ball_b,
        params={
          "racket_cfg": _RACKET_CFG,
          "ball_cfg": _BALL_CFG,
          "robot_cfg": _ROBOT_CFG,
        },
      ),
      "ball_velocity": ObservationTermCfg(
        func=mdp.ball_velocity_b,
        params={"ball_cfg": _BALL_CFG, "robot_cfg": _ROBOT_CFG},
      ),
      "racket_velocity": ObservationTermCfg(
        func=mdp.racket_velocity_b,
        params={"racket_cfg": _RACKET_CFG, "robot_cfg": _ROBOT_CFG},
      ),
      "ball_predicted_landing": ObservationTermCfg(
        func=mdp.ball_predicted_landing_b,
        params={"ball_cfg": _BALL_CFG, "robot_cfg": _ROBOT_CFG},
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

  # -------------------------------------------------------------------------
  # Actions: latent -> frozen decoder -> joint position commands.
  # -------------------------------------------------------------------------
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

  # -------------------------------------------------------------------------
  # Reset events.
  # -------------------------------------------------------------------------
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
      params={
        # Default ballistic feeder: spawn above the net, target the robot's
        # service box on the self side.  Override fields per-robot if needed.
        "provider_cfg": RandomFeederCfg(
          ball_cfg=_BALL_CFG,
          spawn_x_range=(-0.4, 0.4),
          spawn_y_range=(-2.0, 2.0),
          spawn_z_range=(1.0, 1.6),
          target_x_range=(1.5, 4.5),
          target_y_range=(-2.0, 2.0),
          lin_vel_z_range=(1.5, 3.5),
        ),
      },
    ),
    "reset_court": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
        "asset_cfg": _COURT_CFG,
      },
    ),
  }

  # -------------------------------------------------------------------------
  # Racket-ball contact sensor (drives both rewards and terminations).
  # -------------------------------------------------------------------------
  hit_sensor = ContactSensorCfg(
    name=_RACKET_BALL_SENSOR,
    primary=ContactMatch(mode="geom", pattern="tennis_ball", entity="ball"),
    secondary=ContactMatch(
      mode="geom",
      pattern="tennis_racket_collision",
      entity="robot",
    ),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )

  # -------------------------------------------------------------------------
  # Rewards: tiered milestones (1x approach -> 10x hit -> 100x over-net)
  # plus standard regularizing penalties.
  # -------------------------------------------------------------------------
  tracker_params = {
    "sensor_name": _RACKET_BALL_SENSOR,
    "ball_cfg": _BALL_CFG,
    "force_threshold": HIT_FORCE_THRESHOLD,
    "ground_z": GROUND_Z,
    "net_x": NET_X,
  }

  rewards = {
    # --- Goal-driven (tiered) ----------------------------------------------
    "approach_ball": RewardTermCfg(
      func=mdp.racket_to_ball_distance_dense,
      weight=1.0,
      params={
        "std": 0.4,
        "racket_cfg": _RACKET_CFG,
        "ball_cfg": _BALL_CFG,
        "robot_cfg": _ROBOT_CFG,
      },
    ),
    "racket_hit_event": RewardTermCfg(
      func=mdp.racket_hit_event,
      weight=10.0,
      params=dict(tracker_params),
    ),
    "crossed_net_event": RewardTermCfg(
      func=mdp.crossed_net_event,
      weight=100.0,
      params=dict(tracker_params),
    ),
    # --- Survival ----------------------------------------------------------
    "alive": RewardTermCfg(func=mdp.is_alive, weight=0.01),
    # --- Regularization ----------------------------------------------------
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
    # --- Failure penalties (turn termination edges into negative reward) --
    "fall_penalty": RewardTermCfg(
      func=mdp.termination_terms_any,
      weight=-200.0,
      params={"term_names": ("bad_orientation", "root_height")},
    ),
    "ball_out_penalty": RewardTermCfg(
      func=mdp.termination_term,
      weight=-50.0,
      params={"term_name": "ball_out_of_bounds"},
    ),
    "second_contact_penalty": RewardTermCfg(
      func=mdp.termination_term,
      weight=-30.0,
      params={"term_name": "second_contact"},
    ),
  }

  # -------------------------------------------------------------------------
  # Terminations.
  # -------------------------------------------------------------------------
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "nan_detection": TerminationTermCfg(func=mdp.nan_detection),
    # Robot pose failures.
    "bad_orientation": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "root_height": TerminationTermCfg(
      func=mdp.root_height_below_minimum,
      params={"minimum_height": 0.45},
    ),
    # Ball-event terminations (refactored).
    "ball_out_of_bounds": TerminationTermCfg(
      func=mdp.ball_in_play,
      params={
        "x_limits": COURT_OUT_X_LIMITS,
        "y_limits": COURT_OUT_Y_LIMITS,
        "z_limits": COURT_OUT_Z_LIMITS,
      },
    ),
    "second_contact": TerminationTermCfg(
      func=mdp.second_contact, params=dict(tracker_params)
    ),
    "crossed_net_after_hit": TerminationTermCfg(
      func=mdp.crossed_net_after_hit, params=dict(tracker_params)
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
      extent=8.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=6.0,
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
  # Silence "unused import" warnings while keeping public re-exports for
  # robot-specific configs.
  _ = NET_CENTER_HEIGHT
  return cfg
