"""与机器人无关的网球击球任务配置（重构版）。

重构亮点
-------------------
* **球场** : G1 比例网球场，由 :mod:`mjlab.tasks.tennis.scene` 构建
              （总长 14 m × 4.8 m，网中央高 0.52 m）。参见 ``scene.py``。
* **发球逻辑** : :class:`RandomFeederCfg` 弹道轨迹发球器。
              球生成于网上方，机器人侧落点随机采样，
              vz 随机采样，(vx, vy) 由运动学逆解析解确定，
              使球以弧线落入目标区域。
* **终止条件** :
              - 球超出球场边界
              - 球发生第二次接触（球拍或地面）
              - 击球后球成功越过球网
              - 机器人姿态异常 / 倒地
* **奖励** : 接近奖励（1×），球拍击球事件（10×），越网事件（100×），
              以及标准惩罚（关节限位、扭矩、动作变化率等）。
* **观测** :
              - **actor**: 带噪声本体感知 + 10 步球位置窗口
              - **critic**: 完整干净状态（本体感知 + 球/球拍速度 +
                相对向量 + 预测落点）
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
# 场景实体配置（在运行时由场景解析）。
# ---------------------------------------------------------------------------
_ROBOT_CFG = SceneEntityCfg("robot", joint_names=(".*",))
_RACKET_CFG = SceneEntityCfg("robot", site_names=("tennis_racket_center",))
_BALL_CFG = SceneEntityCfg("ball")
_COURT_CFG = SceneEntityCfg("court")
_RACKET_BALL_SENSOR = "racket_ball_contact"

# ---------------------------------------------------------------------------
# 机器人复位（己方半场中部，面向球网）。
# ---------------------------------------------------------------------------
ROBOT_RESET_X_RANGE = (3.5, 4.5)
ROBOT_RESET_Y_RANGE = (-0.4, 0.4)
ROBOT_RESET_YAW = math.pi  # 面向 -x 方向（朝向球网 / 对手）

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
# 遗留常量，保留以向后兼容 Rally / Return 任务（return_env_cfg.py）。
# 重构后的 Hit 任务**不使用**这些常量——其终止逻辑基于更简单的
# TennisRallyTracker（第二次接触 / 越网），而非速度阈值化的"有效击球"概念。
# ---------------------------------------------------------------------------
VALID_HIT_MIN_LEFTWARD_SPEED = 2.0
VALID_HIT_MIN_BALL_SPEED = 2.5
SUCCESS_TARGET_LINE_X = -3.5  # rescaled from -2.2 for the larger court
MISS_BALL_X_OFFSET = 0.2
MISS_BALL_X_DIRECTION = 1.0


def make_tennis_latent_env_cfg() -> ManagerBasedRlEnvCfg:
  """创建重构后的击球任务，使用冻结低层潜变量解码器。

  机器人专属模块负责填充机器人资产、动作缩放和视角主体。
  actor 接收解码器兼容的本体感知（带噪声）以及 10 步球位置窗口；
  critic 接收相同的本体感知（无噪声），并追加球/球拍速度和预测落点。
  """
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

  # ---- Actor：带噪声本体感知 + 10 步球位置窗口 ---
  actor_terms = dict(proprio_actor)
  # 球相对于机器人底部的位置，最近 10 帧（展平后 => 30 维）。
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
  # 动作：潜变量 -> 冻结解码器 -> 关节位置指令。
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
  # 重置事件。
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
        # 默认弹道发球器：在网上方生成球，目标为机器人侧发球区。
        # 如有需要可按机器人类型覆盖各字段。
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
  # 球拍-球接触传感器（驱动奖励与终止条件）。
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
  # 奖励：分层里程碑（1× 接近 -> 10× 击球 -> 100× 越网）
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
    "approach_ball": RewardTermCfg(
      func=mdp.racket_to_ball_distance_dense,
      weight=5.0,
      params={
        "std": 0.4,
        "racket_cfg": _RACKET_CFG,
        "ball_cfg": _BALL_CFG,
        "robot_cfg": _ROBOT_CFG,
      },
    ),
    "racket_hit_event": RewardTermCfg(
      func=mdp.racket_hit_event,
      weight=50.0,
      params=dict(tracker_params),
    ),
    "crossed_net_event": RewardTermCfg(
      func=mdp.crossed_net_event,
      weight=200.0,
      params=dict(tracker_params),
    ),
    # --- 存活奖励 ----------------------------------------------------------
    "alive": RewardTermCfg(func=mdp.is_alive, weight=0.01),
    # --- 正则化惩罚 ----------------------------------------------------
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
    # --- 失败惩罚（将终止边沿转为负奖励）--
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
    # 球事件终止条件（重构版）。
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
