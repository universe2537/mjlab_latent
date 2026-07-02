"""Unitree G1 平地跟踪环境配置。"""

from mjlab.asset_zoo.robots import (
  G1_W_RACKET_ACTION_SCALE,
  get_g1_w_racket_robot_cfg,
)
from mjlab.asset_zoo.robots.unitree_g1_w_pingpong_paddle import (
  get_g1_w_pingpong_paddle_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

G1_TRACKING_BODY_NAMES = (
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

TABLE_TENNIS_TRAIN_MOTION_NAMES = (
  "fanshou_001",
  "fanshou_a_001",
  "fanshou_jingtai_001",
  "fanshou_jingtai_a_001",
  "mix_001",
  "mix_002",
  "mix_a_001",
  "mix_a_002",
  "zhengshou_a_001",
  "zhengshou_a_002",
  "zhengshou_jingtai_001",
)
TABLE_TENNIS_VALIDATION_MOTION_NAMES = ("test_001",)
TABLE_TENNIS_DIAGNOSTIC_MOTION_NAMES = ("zhengshou_002_badend",)
TABLE_TENNIS_TRAIN_MOTION_FILES = tuple(
  f"./artifacts/table_tennis/{name}/motion.npz"
  for name in TABLE_TENNIS_TRAIN_MOTION_NAMES
)


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

  # 自碰撞传感器将 pelvis 子树与自身进行比较。
  # 奖励项随后会根据力的历史记录惩罚重复接触。
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
  # 在训练期间，动作以 W&B artifact 的形式存储。
  # 提供多个 artifact 路径以便命令项可以采样不同的轨迹。
  motion_cmd.motion_source = "local"
  motion_cmd.motion_files = (
    "./artifacts/tennis_random_001/motion.npz",
    "./artifacts/tennis_random_002/motion.npz",
    "./artifacts/tennis_random_003/motion.npz",
    "./artifacts/tennis_random_004/motion.npz",
  )
  motion_cmd.anchor_body_name = "torso_link"
  # 顺序很重要：motion NPZ 张量必须使用相同的身体顺序。
  motion_cmd.body_names = G1_TRACKING_BODY_NAMES

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

  # 如果部署环境未提供状态估计，则移除依赖于全局或根状态（策略无法获得）的观测项。
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

  # Play 模式是确定性且对查看友好的：无扰动、无推力、无观测扰动，重置总是从帧零开始。
  if play:
    # 实际上为无限的 episode 步长。
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # 禁用 RSI 随机化。
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg


def unitree_g1_table_tennis_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create G1 tracking config for table-tennis low-level motions."""
  cfg = unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation,
    play=play,
  )
  cfg.scene.entities = {"robot": get_g1_w_pingpong_paddle_robot_cfg()}

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_source = "local"
  motion_cmd.motion_files = TABLE_TENNIS_TRAIN_MOTION_FILES
  motion_cmd.body_names = G1_TRACKING_BODY_NAMES
  return cfg


__all__ = [
  "G1_TRACKING_BODY_NAMES",
  "TABLE_TENNIS_DIAGNOSTIC_MOTION_NAMES",
  "TABLE_TENNIS_TRAIN_MOTION_FILES",
  "TABLE_TENNIS_TRAIN_MOTION_NAMES",
  "TABLE_TENNIS_VALIDATION_MOTION_NAMES",
  "unitree_g1_flat_tracking_env_cfg",
  "unitree_g1_table_tennis_tracking_env_cfg",
]
