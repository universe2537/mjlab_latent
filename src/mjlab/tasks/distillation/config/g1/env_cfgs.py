"""Unitree G1 latent distillation environment configuration."""

from mjlab.asset_zoo.robots import G1_W_RACKET_ACTION_SCALE, get_g1_w_racket_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.dr import joint
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.distillation import mdp
from mjlab.tasks.distillation.distill_env_cfg import make_distillation_env_cfg
from mjlab.tasks.distillation.mdp import MotionCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

_VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}


def unitree_g1_flat_distillation_env_cfg(has_state_estimation: bool = True,play: bool = False) -> ManagerBasedRlEnvCfg:
  """
  """
  
  cfg = make_distillation_env_cfg()

  cfg.scene.entities = {"robot": get_g1_w_racket_robot_cfg()}
  
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
  motion_cmd.motion_source = "wandb"
  motion_cmd.motion_files = (
    "csv_to_npz/g1_walk1_subject1",
    "csv_to_npz/g1_walk1_subject2",
    "csv_to_npz/g1_walk1_subject5",
    "csv_to_npz/g1_walk2_subject1",
    "csv_to_npz/g1_walk2_subject3",
    "csv_to_npz/g1_walk2_subject4",
    "csv_to_npz/g1_walk3_subject1",
    "csv_to_npz/g1_walk3_subject2",
    "csv_to_npz/g1_walk3_subject3",
    "csv_to_npz/g1_walk3_subject4",
    "csv_to_npz/g1_walk3_subject5",
    "csv_to_npz/g1_walk4_subject1",
    "csv_to_npz/g1_jumps1_subject1",
    "csv_to_npz/g1_jumps1_subject2",
    "csv_to_npz/g1_jumps1_subject5",
    "csv_to_npz/g1_run1_subject2",
    "csv_to_npz/g1_run1_subject5",
    "csv_to_npz/g1_run2_subject1",
    "csv_to_npz/g1_run2_subject4",
    "csv_to_npz/g1_sprint1_subject2",
    "csv_to_npz/g1_sprint1_subject4",
  )
  motion_cmd.anchor_body_name = "torso_link"
  # 顺序很重要：motion NPZ 张量必须使用相同的身体顺序。
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
  # ── Play mode 
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
