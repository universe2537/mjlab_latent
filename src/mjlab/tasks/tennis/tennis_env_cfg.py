"""网球击球任务配置。

重构亮点
-------------------
* **球场** : 标准单打网球场，由 :mod:`mjlab.tasks.tennis.scene` 构建
              （总长 23.77 m × 8.23 m，网中央高 0.914 m）。参见 ``scene.py``。
* **发球逻辑** : :class:`RandomFeederCfg` 弹道轨迹发球器。
              球生成于网上方，机器人侧落点按课程逐步扩展：
              先集中在机器人身上附近，成功率达到 80% 后继续外扩，
              直到覆盖整个己方半场，
              vz 随机采样，(vx, vy) 由运动学逆解析解确定，
              使球以弧线落入目标区域。
* **终止条件** :
              - 球超出球场边界
              - 首次有效击球成功
              - 球首次落地，或再次碰到球拍
              - 机器人姿态异常 / 倒地
* **奖励** : 接近奖励（5×），球拍朝球运动小奖励（1×），
              球拍击球事件（50×），
              以及标准惩罚（关节限位、扭矩、动作变化率等）。
* **观测** :
              - **actor**: 带噪声本体感知 + 10 步球位置窗口 +
                腰部高度预计击球点与 ``time_to_hit``
              - **critic**: 完整干净状态（本体感知 + 球/球拍速度 +
                相对向量 + 预测落点/击球点）
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

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
from mjlab.tasks.tennis import mdp
from mjlab.tasks.tennis.mdp import FrozenDecoderLatentJointPositionActionCfg
from mjlab.tasks.tennis.mdp.ball_providers import RandomFeederCfg
from mjlab.tasks.tennis.mdp.ball_state import OpponentFeederCfg
from mjlab.tasks.tennis.scene import (
  BASELINE_SELF_X,
  COURT_HALF_LENGTH,
  COURT_HALF_WIDTH,
  NET_CENTER_HEIGHT,
  get_tennis_ball_cfg,
  get_tennis_court_cfg,
  get_tennis_terrain_cfg,
  resolve_court_scale,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

# ---------------------------------------------------------------------------
# 场景实体配置（在运行时由场景解析）。
# ---------------------------------------------------------------------------
_ROBOT_CFG = SceneEntityCfg("robot", joint_names=(".*",))
_RACKET_CFG = SceneEntityCfg("robot", site_names=("tennis_racket_center",))
_BALL_CFG = SceneEntityCfg("ball")
_COURT_CFG = SceneEntityCfg("court")
_RACKET_BALL_SENSOR = "racket_ball_contact"
_BALL_NET_SENSOR = "ball_net_contact"

# ---------------------------------------------------------------------------
# 机器人复位（己方半场中后部，面向球网）。
# ---------------------------------------------------------------------------
ROBOT_RESET_X_RANGE = (COURT_HALF_LENGTH * 0.50, COURT_HALF_LENGTH * 0.64)
ROBOT_RESET_Y_RANGE = (-COURT_HALF_WIDTH * 0.17, COURT_HALF_WIDTH * 0.17)
ROBOT_RESET_YAW = math.pi  # 面向 -x 方向（朝向球网 / 对手）

# ---------------------------------------------------------------------------
# 随机发球采样区。落点区间由课程从机器人附近逐步扩展到整个己方半场。
# ---------------------------------------------------------------------------
BALL_SPAWN_X_RANGE = (-0.4, 0.4)
BALL_SPAWN_Y_RANGE = (-COURT_HALF_WIDTH * 0.83, COURT_HALF_WIDTH * 0.83)
BALL_SPAWN_Z_RANGE = (1.0, 1.6)
ROBOT_RESET_X_CENTER = 0.5 * (ROBOT_RESET_X_RANGE[0] + ROBOT_RESET_X_RANGE[1])
BALL_TARGET_INITIAL_X_RANGE = (
  ROBOT_RESET_X_CENTER - 0.15,
  ROBOT_RESET_X_CENTER + 0.15,
)
BALL_TARGET_INITIAL_Y_RANGE = (-0.15, 0.15)
BALL_TARGET_X_RANGE = (0.8, BASELINE_SELF_X - 0.8)
BALL_TARGET_Y_RANGE = (-COURT_HALF_WIDTH, COURT_HALF_WIDTH)
BALL_TARGET_CURRICULUM_SUCCESS_THRESHOLD = 0.8
BALL_TARGET_CURRICULUM_WINDOW = 50
BALL_TARGET_CURRICULUM_STAGES = 6

# ---------------------------------------------------------------------------
# 场景可视化和环境布局参数。
# ---------------------------------------------------------------------------
SCENE_EXTENT = COURT_HALF_LENGTH + 2.0
SCENE_ENV_SPACING = 2.0 * SCENE_EXTENT
VIEWER_DISTANCE = COURT_HALF_LENGTH * 0.9

# ---------------------------------------------------------------------------
# 追踪器阈值。
# ---------------------------------------------------------------------------
HIT_FORCE_THRESHOLD = 1.0
GROUND_Z = 0.06  # 球半径（约 0.034）加小余量
NET_X = 0.0

# ---------------------------------------------------------------------------
# 球出界判定边界（比实际线条略宽松）。
# ---------------------------------------------------------------------------
COURT_OUT_X_LIMITS = (-COURT_HALF_LENGTH - 1.0, BASELINE_SELF_X + 1.0)
COURT_OUT_Y_LIMITS = (-COURT_HALF_WIDTH - 0.5, COURT_HALF_WIDTH + 0.5)
COURT_OUT_Z_LIMITS = (0.02, 3.0)
OPPONENT_LANDING_MARGIN = 0.0
CONTINUOUS_RALLY_SUCCESSFUL_RETURNS = 8
CONTINUOUS_RALLY_INITIAL_SUCCESSFUL_RETURNS = 2
CONTINUOUS_RECOVERY_INITIAL_TIME_RANGE = (3.0, 5.0)
CONTINUOUS_RECOVERY_MID_TIME_RANGE = (1.0, 2.0)
CONTINUOUS_RECOVERY_FINAL_TIME_RANGE = (0.3, 0.5)
CONTINUOUS_RECOVERY_STEPS = 40
CONTINUOUS_RALLY_LENGTH_STAGE_STEPS = (0, 10000 * 24, 25000 * 24)

# ---------------------------------------------------------------------------
# 冻结低层解码器的状态项（必须与蒸馏检查点一致）。
# ---------------------------------------------------------------------------
DECODER_STATE_TERMS = (
  "base_lin_vel",
  "base_ang_vel",
  "joint_pos",
  "joint_vel",
  "actions",
)

# ---------------------------------------------------------------------------
# 球场尺寸类型（供 tyro CLI 自动提示）。
# ---------------------------------------------------------------------------
CourtSizeType = Literal["standard", "half", "mini"]
DEFAULT_COURT_SIZE: CourtSizeType = "half"

# ---------------------------------------------------------------------------
# 预计击球点参数。
# ---------------------------------------------------------------------------
HIT_POINT_HEIGHT_OFFSET = 0.05
HIT_POINT_MAX_HORIZON = 1.5


def _apply_court_geometry(cfg: "TennisLatentEnvCfg") -> None:
  """根据 cfg.court_size 重新计算所有几何相关字段，原地修改 cfg。

  由 ``TennisLatentEnvCfg.__post_init__`` 调用，因此在 tyro 通过 CLI
  覆盖 ``--env.court-size`` 时也会自动触发。
  """
  scale = resolve_court_scale(cfg.court_size)
  cl = COURT_HALF_LENGTH * scale
  cw = COURT_HALF_WIDTH * scale

  robot_reset_x_range = (cl * 0.50, cl * 0.64)
  robot_reset_y_range = (-cw * 0.17, cw * 0.17)
  robot_reset_x_center = 0.5 * (robot_reset_x_range[0] + robot_reset_x_range[1])

  ball_spawn_y_range = (-cw * 0.83, cw * 0.83)
  ball_target_initial_x_range = (
    robot_reset_x_center - 0.15 * scale,
    robot_reset_x_center + 0.15 * scale,
  )
  ball_target_initial_y_range = (-0.15 * scale, 0.15 * scale)
  ball_target_x_range = (
    max(0.3, 0.8 * scale),
    max(0.5, cl - 0.8 * scale),
  )
  ball_target_y_range = (-cw, cw)
  court_out_x_limits = (-cl - 1.0, cl + 1.0)
  court_out_y_limits = (-cw - 0.5, cw + 0.5)
  landing_x_limits = (-cl - OPPONENT_LANDING_MARGIN, NET_X)
  landing_y_limits = (-cw - OPPONENT_LANDING_MARGIN, cw + OPPONENT_LANDING_MARGIN)

  # -- 场景 --
  cfg.scene.entities["court"] = get_tennis_court_cfg(scale=scale)
  cfg.scene.env_spacing = 2.0 * (cl + 2.0)
  cfg.scene.extent = cl + 2.0

  # -- 视角 --
  cfg.viewer.distance = max(3.0, cl * 0.9)

  # -- 机器人复位 --
  cfg.events["reset_robot_base"].params["pose_range"]["x"] = robot_reset_x_range
  cfg.events["reset_robot_base"].params["pose_range"]["y"] = robot_reset_y_range

  # -- 出界判定 --
  ball_out_cfg = cfg.terminations.get("ball_out_of_bounds")
  if ball_out_cfg is not None:
    ball_out_cfg.params["x_limits"] = court_out_x_limits
    ball_out_cfg.params["y_limits"] = court_out_y_limits
    ball_out_params = ball_out_cfg.params
    if "sensor_name" in ball_out_params:
      ball_out_params["landing_x_limits"] = landing_x_limits
      ball_out_params["landing_y_limits"] = landing_y_limits
  for term_name in (
    "crossed_net_event",
    "landing_in_bounds_event",
    "landing_in_bounds_after_hit",
    "second_contact",
    "continuous_rally_failure",
    "continuous_rally_complete",
    "continuous_recovery_ready_pose",
    "respawn_successful_continuous_rally_ball",
    "advance_continuous_rally_ball",
    "continuous_ball_fault",
    "continuous_rally_complete_state",
    "continuous_racket_to_predicted_hit_point_dense",
    "continuous_racket_towards_ball_velocity",
    "continuous_racket_hit_event",
    "continuous_crossed_net_event",
    "continuous_landing_in_bounds_event",
    "continuous_post_hit_x_progress",
    "continuous_post_hit_ball_velocity_direction",
  ):
    term_cfg = cfg.rewards.get(term_name) or cfg.terminations.get(term_name)
    if term_cfg is None:
      continue
    term_cfg.params["landing_x_limits"] = landing_x_limits
    term_cfg.params["landing_y_limits"] = landing_y_limits

  for term_cfg in cfg.metrics.values():
    term_cfg.params["landing_x_limits"] = landing_x_limits
    term_cfg.params["landing_y_limits"] = landing_y_limits

  for obs_group in cfg.observations.values():
    term_cfg = obs_group.terms.get("continuous_rally_phase")
    if term_cfg is None:
      term_cfg = obs_group.terms.get("continuous_ball_phase")
    if term_cfg is None:
      continue
    term_cfg.params["landing_x_limits"] = landing_x_limits
    term_cfg.params["landing_y_limits"] = landing_y_limits

  # -- 球发球器（reset_ball 与课程共享同一对象）--
  ball_provider_cfg = RandomFeederCfg(
    ball_cfg=_BALL_CFG,
    spawn_x_range=BALL_SPAWN_X_RANGE,
    spawn_y_range=ball_spawn_y_range,
    spawn_z_range=BALL_SPAWN_Z_RANGE,
    target_x_range=ball_target_initial_x_range,
    target_y_range=ball_target_initial_y_range,
    lin_vel_z_range=(1.5, 3.5),
  )
  cfg.events["reset_ball"].params["provider_cfg"] = ball_provider_cfg

  # -- 课程 --
  cp = cfg.curriculum["ball_target_region"].params
  cp["provider_cfg"] = ball_provider_cfg
  cp["initial_target_x_range"] = ball_target_initial_x_range
  cp["initial_target_y_range"] = ball_target_initial_y_range
  cp["final_target_x_range"] = ball_target_x_range
  cp["final_target_y_range"] = ball_target_y_range


@dataclass(kw_only=True)
class TennisLatentEnvCfg(ManagerBasedRlEnvCfg):
  """带运行时球场尺寸参数的网球潜变量环境配置。

  在训练或推演启动时通过 ``--env.court-size <size>`` 传入球场大小，
  无需使用不同的任务名称。

  示例::

    uv run train Mjlab-Tennis-Hit-Unitree-G1                   # 默认 mini
    uv run train Mjlab-Tennis-Hit-Unitree-G1 --env.court-size standard
  """

  court_size: CourtSizeType = DEFAULT_COURT_SIZE
  """球场尺寸预设，默认 ``"mini"``。

  可选值：``"standard"`` / ``"half"`` / ``"mini"``。
  """

  def __post_init__(self) -> None:
    _apply_court_geometry(self)


def make_tennis_latent_env_cfg(
  court_size: CourtSizeType = DEFAULT_COURT_SIZE,
) -> TennisLatentEnvCfg:
  """创建重构后的击球任务，使用冻结低层潜变量解码器。

  机器人专属模块负责填充机器人资产、动作缩放和视角主体。
  actor 接收解码器兼容的本体感知（带噪声）以及 10 步球位置窗口；
  critic 接收相同的本体感知（无噪声），并追加球/球拍速度和预测落点。

  参数:
    court_size: 球场尺寸预设，默认 ``"mini"``。可选值见 :data:`CourtSizeType`。
  """
  #
  # 根据 court_size 计算当前尺寸下的几何参数。
  #
  scale = resolve_court_scale(court_size)
  cl = COURT_HALF_LENGTH * scale  # 半场长度
  cw = COURT_HALF_WIDTH * scale  # 半场宽度
  baseline_self_x = cl  # 己方底线 x

  # 机器人复位区（己方半场中后部）。
  robot_reset_x_range = (cl * 0.50, cl * 0.64)
  robot_reset_y_range = (-cw * 0.17, cw * 0.17)
  robot_reset_x_center = 0.5 * (robot_reset_x_range[0] + robot_reset_x_range[1])

  # 球发球生成区。
  ball_spawn_y_range = (-cw * 0.83, cw * 0.83)

  # 落点课程范围（从机器人附近逐步扩展到全半场）。
  ball_target_initial_x_range = (
    robot_reset_x_center - 0.15 * scale,
    robot_reset_x_center + 0.15 * scale,
  )
  ball_target_initial_y_range = (-0.15 * scale, 0.15 * scale)
  ball_target_x_range = (
    max(0.3, 0.8 * scale),
    max(0.5, baseline_self_x - 0.8 * scale),
  )
  ball_target_y_range = (-cw, cw)

  # 球场出界判定边界（比实际线条略宽松）。
  court_out_x_limits = (-cl - 1.0, baseline_self_x + 1.0)
  court_out_y_limits = (-cw - 0.5, cw + 0.5)

  # 场景间距。
  scene_extent = cl + 2.0
  scene_env_spacing = 2.0 * scene_extent
  viewer_distance = max(3.0, cl * 0.9)
  # -------------------------------------------------------------------------
  # 解码器兼容的本体感知（actor 与 critic 共用）。
  # 动作项将从 actor 观测中精确切出 DECODER_STATE_TERMS。
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

  # ---- Actor：带噪声本体感知 + 10 步球位置窗口 + 预计击球点 ---
  actor_terms = dict(proprio_actor)
  # 球相对于球拍中心的位置，最近 10 帧（展平后 => 30 维）。
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

  # ---- Critic：干净本体感知 + 尽可能完整的状态信息 ----------
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

  #
  # 动作：潜变量 -> 冻结解码器 -> 关节位置指令。
  #
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

  ball_provider_cfg = RandomFeederCfg(
    ball_cfg=_BALL_CFG,
    spawn_x_range=BALL_SPAWN_X_RANGE,
    spawn_y_range=ball_spawn_y_range,
    spawn_z_range=BALL_SPAWN_Z_RANGE,
    target_x_range=ball_target_initial_x_range,
    target_y_range=ball_target_initial_y_range,
    lin_vel_z_range=(1.5, 3.5),
  )

  #
  # 重置事件。
  #
  events = {
    "reset_robot_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": robot_reset_x_range,
          "y": robot_reset_y_range,
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
        # 默认弹道发球器：在网上方生成球，目标区由课程逐步扩展。
        # 如有需要可按机器人类型覆盖各字段。
        "provider_cfg": ball_provider_cfg,
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

  #
  # 球拍-球接触传感器（驱动奖励与终止条件）。
  #
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
  # 奖励：分层里程碑（接近 -> 击球）
  # 加上标准正则化惩罚。
  # -------------------------------------------------------------------------
  tracker_params = {
    "sensor_name": _RACKET_BALL_SENSOR,
    "ball_cfg": _BALL_CFG,
    "force_threshold": HIT_FORCE_THRESHOLD,
    "ground_z": GROUND_Z,
    "net_x": NET_X,
  }

  rewards = {
    # --- 目标驱动（分层）----------------------------------------------
    "approach_point": RewardTermCfg(
      func=mdp.racket_to_predicted_hit_point_dense,
      weight=10,
      params={
        "std": 0.4,
        "racket_cfg": _RACKET_CFG,
        "ball_cfg": _BALL_CFG,
        "robot_cfg": _ROBOT_CFG,
        "hit_height_offset": HIT_POINT_HEIGHT_OFFSET,
        "max_horizon": HIT_POINT_MAX_HORIZON,
      },
    ),
    "racket_towards_ball": RewardTermCfg(
      func=mdp.racket_towards_ball_velocity,
      weight=5,
      params={
        **dict(tracker_params),
        "racket_cfg": _RACKET_CFG,
        "ball_cfg": _BALL_CFG,
        "robot_cfg": _ROBOT_CFG,
        "speed_scale": 2.0,
        "distance_std": 0.8,
      },
    ),
    "racket_hit_event": RewardTermCfg(
      func=mdp.racket_hit_event,
      weight=100.0,
      params=dict(tracker_params),
    ),
    # --- 存活奖励 ----------------------------------------------------------
    "alive": RewardTermCfg(func=mdp.is_alive, weight=0.01),
    # --- 正则化惩罚 ----------------------------------------------------
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
    # --- 失败惩罚 --
    "fall_penalty": RewardTermCfg(
      func=mdp.termination_terms_any,
      weight=-200.0,
      params={"term_names": ("bad_orientation", "root_height")},
    ),
  }

  # -------------------------------------------------------------------------
  # 终止条件。
  # -------------------------------------------------------------------------
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "nan_detection": TerminationTermCfg(func=mdp.nan_detection),
    # 机器人姿态失败。
    "bad_orientation": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "root_height": TerminationTermCfg(
      func=mdp.root_height_below_minimum,
      params={"minimum_height": 0.45},
    ),
    # 球事件终止条件。
    "ball_out_of_bounds": TerminationTermCfg(
      func=mdp.ball_in_play,
      params={
        "x_limits": court_out_x_limits,
        "y_limits": court_out_y_limits,
        "z_limits": COURT_OUT_Z_LIMITS,
      },
    ),
    "first_racket_hit": TerminationTermCfg(
      func=mdp.first_racket_hit, params=dict(tracker_params)
    ),
    "second_contact": TerminationTermCfg(
      func=mdp.second_contact, params=dict(tracker_params)
    ),
  }

  curriculum = {
    "ball_target_region": CurriculumTermCfg(
      func=mdp.random_feeder_target_curriculum,
      params={
        "provider_cfg": ball_provider_cfg,
        "initial_target_x_range": ball_target_initial_x_range,
        "initial_target_y_range": ball_target_initial_y_range,
        "final_target_x_range": ball_target_x_range,
        "final_target_y_range": ball_target_y_range,
        "success_term_name": "first_racket_hit",
        "success_threshold": BALL_TARGET_CURRICULUM_SUCCESS_THRESHOLD,
        "success_window": BALL_TARGET_CURRICULUM_WINDOW,
        "num_stages": BALL_TARGET_CURRICULUM_STAGES,
      },
    )
  }

  metrics = {
    "racket_hit_count": MetricsTermCfg(
      func=mdp.racket_hit_count_metric,
      reduce="last",
      params=dict(tracker_params),
    ),
    "crossed_net_count": MetricsTermCfg(
      func=mdp.crossed_net_count_metric,
      reduce="last",
      params=dict(tracker_params),
    ),
    "landing_in_bounds_count": MetricsTermCfg(
      func=mdp.landing_in_bounds_count_metric,
      reduce="last",
      params={
        **dict(tracker_params),
        "landing_x_limits": None,
        "landing_y_limits": None,
      },
    ),
    "successful_return_count": MetricsTermCfg(
      func=mdp.successful_return_count_metric,
      reduce="last",
      params={
        **dict(tracker_params),
        "landing_x_limits": None,
        "landing_y_limits": None,
      },
    ),
  }

  cfg = TennisLatentEnvCfg(
    court_size=court_size,
    scene=SceneCfg(
      terrain=get_tennis_terrain_cfg(),
      entities={
        "ball": get_tennis_ball_cfg(),
        "court": get_tennis_court_cfg(scale=scale),
      },
      sensors=(hit_sensor,),
      num_envs=1,
      env_spacing=scene_env_spacing,
      extent=scene_extent,
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
      distance=viewer_distance,
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


def make_tennis_latent_cross_env_cfg(
  court_size: CourtSizeType = DEFAULT_COURT_SIZE,
) -> TennisLatentEnvCfg:
  """创建击球过网并落在对方界内的潜变量网球任务。"""
  cfg = make_tennis_latent_env_cfg(court_size=court_size)
  scale = resolve_court_scale(court_size)
  cl = COURT_HALF_LENGTH * scale
  cw = COURT_HALF_WIDTH * scale
  landing_x_limits = (-cl - OPPONENT_LANDING_MARGIN, NET_X)
  landing_y_limits = (-cw - OPPONENT_LANDING_MARGIN, cw + OPPONENT_LANDING_MARGIN)

  tracker_params = {
    "sensor_name": _RACKET_BALL_SENSOR,
    "ball_cfg": _BALL_CFG,
    "force_threshold": HIT_FORCE_THRESHOLD,
    "ground_z": GROUND_Z,
    "net_x": NET_X,
    "landing_x_limits": landing_x_limits,
    "landing_y_limits": landing_y_limits,
  }

  cfg.rewards["racket_towards_ball"].params.update(
    {
      "landing_x_limits": landing_x_limits,
      "landing_y_limits": landing_y_limits,
    }
  )
  cfg.rewards["racket_hit_event"].params.update(
    {
      "landing_x_limits": landing_x_limits,
      "landing_y_limits": landing_y_limits,
    }
  )
  for term_cfg in cfg.metrics.values():
    term_cfg.params["landing_x_limits"] = landing_x_limits
    term_cfg.params["landing_y_limits"] = landing_y_limits
  cfg.rewards["approach_point"].weight = 5.0
  cfg.rewards["racket_towards_ball"].weight = 2.0
  cfg.rewards["racket_hit_event"].weight = 25.0
  cfg.rewards["post_hit_x_progress"] = RewardTermCfg(
    func=mdp.post_hit_x_progress,
    weight=50.0,
    params={**dict(tracker_params), "max_progress": 0.05},
  )
  cfg.rewards["post_hit_ball_velocity_direction"] = RewardTermCfg(
    func=mdp.post_hit_ball_velocity_direction,
    weight=20.0,
    params={
      **dict(tracker_params),
      "x_speed_scale": 4.0,
      "lateral_speed_std": 1.5,
    },
  )
  cfg.rewards["crossed_net_event"] = RewardTermCfg(
    func=mdp.crossed_net_event,
    weight=500.0,
    params=dict(tracker_params),
  )
  cfg.rewards["landing_in_bounds_event"] = RewardTermCfg(
    func=mdp.landing_in_bounds_event,
    weight=1000.0,
    params=dict(tracker_params),
  )

  cfg.terminations.pop("first_racket_hit", None)
  cfg.terminations["landing_in_bounds_after_hit"] = TerminationTermCfg(
    func=mdp.landing_in_bounds_after_hit,
    params=dict(tracker_params),
  )
  cfg.terminations["second_contact"].params.update(
    {
      "landing_x_limits": landing_x_limits,
      "landing_y_limits": landing_y_limits,
    }
  )
  cfg.curriculum["ball_target_region"].params["success_term_name"] = (
    "landing_in_bounds_after_hit"
  )
  return cfg


def make_tennis_continuous_env_cfg(
  court_size: CourtSizeType = DEFAULT_COURT_SIZE,
  max_successful_returns: int = CONTINUOUS_RALLY_SUCCESSFUL_RETURNS,
) -> TennisLatentEnvCfg:
  """创建连续接多球任务。

  与 Cross 不同，成功把球打过网并落在对方界内不会立即结束 episode；
  环境会重新随机发球并进入下一小回合。连续成功 ``max_successful_returns``
  次后才作为整局成功终止。
  """
  cfg = make_tennis_latent_cross_env_cfg(court_size=court_size)
  scale = resolve_court_scale(court_size)
  cl = COURT_HALF_LENGTH * scale
  cw = COURT_HALF_WIDTH * scale
  robot_reset_x_range = (cl * 0.50, cl * 0.64)
  robot_reset_x_center = 0.5 * (robot_reset_x_range[0] + robot_reset_x_range[1])
  initial_successful_returns = min(
    CONTINUOUS_RALLY_INITIAL_SUCCESSFUL_RETURNS,
    max_successful_returns,
  )
  landing_x_limits = (-cl - OPPONENT_LANDING_MARGIN, NET_X)
  landing_y_limits = (-cw - OPPONENT_LANDING_MARGIN, cw + OPPONENT_LANDING_MARGIN)
  court_out_x_limits = (-cl - 1.0, cl + 1.0)
  court_out_y_limits = (-cw - 0.5, cw + 0.5)
  continuous_z_limits = (COURT_OUT_Z_LIMITS[0], 4.0)
  continuous_params = {
    "racket_sensor_name": _RACKET_BALL_SENSOR,
    "net_sensor_name": _BALL_NET_SENSOR,
    "ball_cfg": _BALL_CFG,
    "force_threshold": HIT_FORCE_THRESHOLD,
    "ground_z": GROUND_Z,
    "net_x": NET_X,
    "net_height": NET_CENTER_HEIGHT,
    "landing_x_limits": landing_x_limits,
    "landing_y_limits": landing_y_limits,
    "x_limits": court_out_x_limits,
    "y_limits": court_out_y_limits,
    "z_limits": continuous_z_limits,
  }
  target_initial_x_range = (
    robot_reset_x_center - 0.15 * scale,
    robot_reset_x_center + 0.15 * scale,
  )
  target_initial_y_range = (-0.15 * scale, 0.15 * scale)
  target_x_range = (max(0.3, 0.8 * scale), max(0.5, cl - 0.8 * scale))
  target_y_range = (-cw, cw)
  phase_params = {
    **dict(continuous_params),
    "max_successful_returns": max_successful_returns,
  }
  opponent_provider_cfg = OpponentFeederCfg(
    ball_cfg=_BALL_CFG,
    spawn_x_range=(-cl + 0.2 * scale, -max(0.2, 0.3 * scale)),
    spawn_y_range=(-cw, cw),
    target_x_range=target_initial_x_range,
    target_y_range=target_initial_y_range,
    flight_time_range=(0.85, 1.35),
    flight_time_slack_range=(0.05, 0.35),
    spawn_z_range=(GROUND_Z, GROUND_Z),
    ground_z=GROUND_Z,
    net_x=NET_X,
    net_height=NET_CENTER_HEIGHT,
    net_clearance=0.25,
  )

  ball_net_sensor = ContactSensorCfg(
    name=_BALL_NET_SENSOR,
    primary=ContactMatch(mode="geom", pattern="tennis_ball", entity="ball"),
    secondary=ContactMatch(
      mode="geom",
      pattern="tennis_net_collision",
      entity="court",
    ),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (*cfg.scene.sensors, ball_net_sensor)
  cfg.events["reset_ball"].params["provider_cfg"] = opponent_provider_cfg

  cfg.episode_length_s = 20.0
  cfg.observations["actor"].terms["continuous_ball_phase"] = ObservationTermCfg(
    func=mdp.continuous_ball_phase,
    params=dict(phase_params),
  )
  cfg.observations["critic"].terms["continuous_ball_phase"] = ObservationTermCfg(
    func=mdp.continuous_ball_phase,
    params=dict(phase_params),
  )
  cfg.rewards[
    "approach_point"
  ].func = mdp.continuous_racket_to_predicted_hit_point_dense
  cfg.rewards["approach_point"].params = {
    **dict(continuous_params),
    "std": 0.4,
    "racket_cfg": _RACKET_CFG,
    "robot_cfg": _ROBOT_CFG,
    "hit_height_offset": HIT_POINT_HEIGHT_OFFSET,
    "max_horizon": HIT_POINT_MAX_HORIZON,
  }
  cfg.rewards["racket_towards_ball"].func = mdp.continuous_racket_towards_ball_velocity
  cfg.rewards["racket_towards_ball"].params = {
    **dict(continuous_params),
    "racket_cfg": _RACKET_CFG,
    "robot_cfg": _ROBOT_CFG,
    "speed_scale": 2.0,
    "distance_std": 0.8,
  }
  cfg.rewards["racket_hit_event"].func = mdp.continuous_racket_hit_event
  cfg.rewards["racket_hit_event"].params = dict(continuous_params)
  cfg.rewards["post_hit_x_progress"].func = mdp.continuous_post_hit_x_progress
  cfg.rewards["post_hit_x_progress"].params = {
    **dict(continuous_params),
    "max_progress": 0.05,
  }
  cfg.rewards[
    "post_hit_ball_velocity_direction"
  ].func = mdp.continuous_post_hit_ball_velocity_direction
  cfg.rewards["post_hit_ball_velocity_direction"].params = {
    **dict(continuous_params),
    "x_speed_scale": 4.0,
    "lateral_speed_std": 1.5,
  }
  cfg.rewards["crossed_net_event"].func = mdp.continuous_crossed_net_event
  cfg.rewards["crossed_net_event"].params = dict(continuous_params)
  cfg.rewards["landing_in_bounds_event"].func = mdp.continuous_landing_in_bounds_event
  cfg.rewards["landing_in_bounds_event"].params = dict(continuous_params)

  cfg.terminations.pop("ball_out_of_bounds", None)
  cfg.terminations.pop("landing_in_bounds_after_hit", None)
  cfg.terminations.pop("second_contact", None)
  cfg.terminations["continuous_ball_fault"] = TerminationTermCfg(
    func=mdp.continuous_ball_fault,
    params=dict(continuous_params),
  )
  cfg.terminations["continuous_rally_complete"] = TerminationTermCfg(
    func=mdp.continuous_rally_complete_state,
    params={
      **dict(continuous_params),
      "max_successful_returns": initial_successful_returns,
    },
  )

  cfg.rewards["continuous_rally_complete_bonus"] = RewardTermCfg(
    func=mdp.termination_terms_any,
    weight=2000.0,
    params={"term_names": ("continuous_rally_complete",)},
  )
  cfg.rewards["continuous_recovery_ready_pose"] = RewardTermCfg(
    func=mdp.continuous_recovery_ready_pose_state,
    weight=20.0,
    params={
      **dict(continuous_params),
      "racket_cfg": _RACKET_CFG,
      "robot_cfg": _ROBOT_CFG,
      "target_x": robot_reset_x_center,
      "target_y": 0.0,
      "target_heading": ROBOT_RESET_YAW,
    },
  )
  cfg.rewards["advance_continuous_rally_ball"] = RewardTermCfg(
    func=mdp.advance_continuous_rally_ball,
    weight=1.0e-9,
    params={
      **dict(continuous_params),
      "provider_cfg": opponent_provider_cfg,
      "max_successful_returns": initial_successful_returns,
      "recovery_time_range": CONTINUOUS_RECOVERY_INITIAL_TIME_RANGE,
    },
  )
  cfg.rewards.pop("respawn_successful_continuous_rally_ball", None)
  cfg.metrics["racket_hit_count"].func = mdp.continuous_racket_hit_count_metric
  cfg.metrics["racket_hit_count"].params = dict(continuous_params)
  cfg.metrics["crossed_net_count"].func = mdp.continuous_crossed_net_count_metric
  cfg.metrics["crossed_net_count"].params = dict(continuous_params)
  cfg.metrics[
    "landing_in_bounds_count"
  ].func = mdp.continuous_landing_in_bounds_count_metric
  cfg.metrics["landing_in_bounds_count"].params = dict(continuous_params)
  cfg.metrics[
    "successful_return_count"
  ].func = mdp.continuous_successful_return_count_metric
  cfg.metrics["successful_return_count"].params = dict(continuous_params)
  cfg.metrics["continuous_success_ratio"] = MetricsTermCfg(
    func=mdp.continuous_success_ratio_metric_state,
    reduce="last",
    params={
      **dict(continuous_params),
      "max_successful_returns": max_successful_returns,
    },
  )
  cfg.metrics["in_recovery_rate"] = MetricsTermCfg(
    func=mdp.continuous_in_recovery_metric_state,
    params=dict(continuous_params),
  )
  cfg.metrics["net_contact_count"] = MetricsTermCfg(
    func=mdp.continuous_net_contact_count_metric,
    reduce="last",
    params=dict(continuous_params),
  )
  cfg.metrics["invalid_feed_count"] = MetricsTermCfg(
    func=mdp.continuous_invalid_feed_count_metric,
    reduce="last",
    params=dict(continuous_params),
  )
  cfg.metrics["continuous_fault_count"] = MetricsTermCfg(
    func=mdp.continuous_fault_count_metric,
    reduce="last",
    params=dict(continuous_params),
  )

  cfg.curriculum["ball_target_region"].params["success_term_name"] = (
    "continuous_rally_complete"
  )
  cfg.curriculum["ball_target_region"].params["provider_cfg"] = opponent_provider_cfg
  cfg.curriculum["ball_target_region"].params["initial_target_x_range"] = (
    target_initial_x_range
  )
  cfg.curriculum["ball_target_region"].params["initial_target_y_range"] = (
    target_initial_y_range
  )
  cfg.curriculum["ball_target_region"].params["final_target_x_range"] = target_x_range
  cfg.curriculum["ball_target_region"].params["final_target_y_range"] = target_y_range
  rally_length_stages = [
    {
      "step": CONTINUOUS_RALLY_LENGTH_STAGE_STEPS[0],
      "params": {"max_successful_returns": initial_successful_returns},
    },
    {
      "step": CONTINUOUS_RALLY_LENGTH_STAGE_STEPS[1],
      "params": {"max_successful_returns": min(4, max_successful_returns)},
    },
    {
      "step": CONTINUOUS_RALLY_LENGTH_STAGE_STEPS[2],
      "params": {"max_successful_returns": max_successful_returns},
    },
  ]
  cfg.curriculum["continuous_rally_length"] = CurriculumTermCfg(
    func=mdp.termination_curriculum,
    params={
      "termination_name": "continuous_rally_complete",
      "stages": rally_length_stages,
    },
  )
  cfg.curriculum["continuous_respawn_length"] = CurriculumTermCfg(
    func=mdp.reward_curriculum,
    params={
      "reward_name": "advance_continuous_rally_ball",
      "stages": rally_length_stages,
    },
  )
  wait_interval_stages = [
    {
      "step": CONTINUOUS_RALLY_LENGTH_STAGE_STEPS[0],
      "params": {"recovery_time_range": CONTINUOUS_RECOVERY_INITIAL_TIME_RANGE},
    },
    {
      "step": CONTINUOUS_RALLY_LENGTH_STAGE_STEPS[1],
      "params": {"recovery_time_range": CONTINUOUS_RECOVERY_MID_TIME_RANGE},
    },
    {
      "step": CONTINUOUS_RALLY_LENGTH_STAGE_STEPS[2],
      "params": {"recovery_time_range": CONTINUOUS_RECOVERY_FINAL_TIME_RANGE},
    },
  ]
  cfg.curriculum["continuous_wait_interval"] = CurriculumTermCfg(
    func=mdp.reward_curriculum,
    params={
      "reward_name": "advance_continuous_rally_ball",
      "stages": wait_interval_stages,
    },
  )
  return cfg
