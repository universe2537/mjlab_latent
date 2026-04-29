"""Unitree G1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_W_RACKET_ACTION_SCALE,
  get_g1_w_racket_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat-terrain tracking configuration.

  This specializes the generic tracking task with G1-specific action scaling,
  body names, contact sensors, and default motion artifacts.
  """
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_g1_w_racket_robot_cfg()}

  # The self-collision sensor compares the pelvis subtree with itself.  The
  # reward term later interprets force history to penalize repeated contacts.
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_W_RACKET_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  # Motions are stored as W&B artifacts during training.  Multiple artifact
  # paths are provided so the command term can sample different trajectories.
  motion_cmd.motion_source = "wandb"
  motion_cmd.motion_files = (
    "csv_to_npz/g1_dance1_subject1",
    "csv_to_npz/g1_dance1_subject2",
    "csv_to_npz/g1_dance1_subject3",
    "csv_to_npz/g1_dance2_subject1",
    "csv_to_npz/g1_dance2_subject2",
    "csv_to_npz/g1_dance2_subject3",
    "csv_to_npz/g1_dance2_subject4",
    "csv_to_npz/g1_dance2_subject5",
  )
  motion_cmd.anchor_body_name = "torso_link"
  # Order matters: the motion NPZ tensors must use the same body order.
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  # If deployment does not provide state estimation, remove observations that
  # depend on global/root state not available to the policy.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Play mode is deterministic and viewer-friendly: no perturbations, no pushes,
  # no observation corruption, and a reset always starts from frame zero.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg
