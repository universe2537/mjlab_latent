"""Unitree Go1 velocity environment configurations."""

import math
from typing import Literal

from mjlab.asset_zoo.robots import (
  GO1_ACTION_SCALE,
  get_go1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.config import pyramid_stairs, pyramid_stairs_inv
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

TerrainType = Literal["rough", "obstacles"]


def _step_height_for_peak_height(
  terrain_size: tuple[float, float],
  border_width: float,
  platform_width: float,
  step_width: float,
  peak_height: float,
) -> float:
  """将目标金字塔顶部峰值高度转换为单步台阶高度参数。

  金字塔台阶地形由多层等高阶梯构成，给定整个金字塔希望达到的
  最大高度（peak_height），反推每一级台阶应设多高。

  Args:
    terrain_size:  整块地形的 (宽, 长)，单位米。
    border_width:  地形四周平坦边框的宽度，单位米。
    platform_width: 金字塔顶部平台的边长，单位米。
    step_width:    每一级台阶的水平深度（run），单位米。
    peak_height:   希望台阶最高点相对地面的总高度，单位米。

  Returns:
    传给 SubTerrainCfg.step_height_range 使用的单步高度，单位米。
  """
  # 去掉两侧边框和顶部平台后，可用于放置台阶的水平长度
  usable_x = terrain_size[0] - 2.0 * border_width - platform_width
  usable_y = terrain_size[1] - 2.0 * border_width - platform_width
  # 实际能放下多少级台阶（双侧对称，每侧占 step_width，故除以 2）
  num_steps = max(1, int(min(usable_x, usable_y) / (2.0 * step_width)))
  # 单步高度 = 总峰值高度 / (台阶数 + 1)，+1 是因为顶层平台也贡献一份高度
  return peak_height / (num_steps + 1)


def unitree_go1_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """创建 Unitree Go1 粗糙地形速度跟踪训练配置。

  以通用速度任务配置为基础，叠加 Go1 专属的传感器、奖励、
  碰撞检测和地形课程设置。当 play=True 时额外关闭噪声、
  无限延长 episode，并切换到随机地形模式供可视化使用。

  Args:
    play: 若为 True，则应用 play 模式覆盖（关闭 curriculum、
          禁用干扰推力、无限 episode 时长等）。
  """
  # 从通用速度任务模板创建基础配置（包含通用奖励、命令、终止条件等）
  cfg = make_velocity_env_cfg()

  # CCD（连续碰撞检测）最大迭代次数。
  # Go1 腿部几何体较复杂，设为 500 次确保台阶/斜坡上的碰撞精度，
  # 代价是每步仿真计算量增大。
  cfg.sim.mujoco.ccd_iterations = 500
  # 冲力比（impedance ratio）。值越大接触越硬（弹性越小），
  # 10 可防止脚部在台阶边缘出现"穿透"现象，但会略微降低仿真稳定性。
  cfg.sim.mujoco.impratio = 10
  # 摩擦锥模型："elliptic" 为椭圆锥（各向异性），比默认的金字塔锥
  # 在侧向力建模上更准确，适合有横向滑动的复杂地形。
  cfg.sim.mujoco.cone = "elliptic"
  # 接触传感器允许的最大接触匹配数。
  # Go1 有多组接触传感器（脚底、大腿、小腿、躯干），设为 500
  # 以防止密集接触场景（如台阶边缘多点接触）时丢失接触事件。
  cfg.sim.contact_sensor_maxmatch = 500

  # 将 Go1 机器人模型注入场景，key "robot" 是后续所有
  # SceneEntityCfg("robot", ...) 引用的名称。
  cfg.scene.entities = {"robot": get_go1_robot_cfg()}

  # terrain_scan 是向下发射射线的高度场传感器，用于感知脚下地形轮廓。
  # 基础模板默认绑定到 G1 的躯干，这里改绑到 Go1 的 "trunk" body。
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "trunk"  # Go1 的主躯干 body 名称

  # Go1 四条腿的末端命名规则：FR=右前, FL=左前, RR=右后, RL=左后
  foot_names = ("FR", "FL", "RR", "RL")
  # MuJoCo 模型中每个足端的 site 名称，用于绑定高度传感器采样原点
  site_names = ("FR", "FL", "RR", "RL")
  # 每个足端对应的碰撞几何体名称，用于接触传感器和摩擦随机化
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  # foot_height_scan 传感器为每个足端独立采样脚下地形高度，
  # 用于计算足部离地高度（foot clearance）奖励。
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      # 将采样原点绑定到每个足端的 site（world frame 位置随仿真更新）
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      # 以足端为圆心、半径 0.04m 的单圆环，均匀分布 4 个采样点。
      # 采样半径接近足底半径，可准确判断台阶边缘的支撑情况。
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

  # ── 接触传感器：足端落地检测 ─────────────────────────────────────────
  # 检测四个足端碰撞几何体与地形 body 之间的接触状态。
  # fields=("found", "force"): 同时记录是否接触 + 接触力大小。
  # reduce="netforce": 将同一足端的多个接触点合并为一个合力向量，
  #   避免台阶边缘多点接触时数据冗余。
  # num_slots=1: 每个足端只保存最近 1 步的接触状态（节省显存）。
  # track_air_time=True: 自动统计每个足端的离地时长，用于 air_time 奖励。
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  # ── 接触传感器：躯干自碰撞检测 ───────────────────────────────────────
  # primary 和 secondary 都是 trunk 子树，检测机器人自身肢体相互碰撞。
  # reduce="none": 保留每个接触对的独立信息（不合并），方便精确惩罚。
  # history_length=4: 保存最近 4 步历史，防止单帧漏检瞬时碰撞。
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  # 每条腿有 3 个大腿碰撞几何体（thigh_collision1/2/3），共 12 个
  thigh_geom_names = tuple(
    f"{leg}_thigh_collision{i}" for leg in foot_names for i in (1, 2, 3)
  )

  # ── 接触传感器：大腿触地检测 ─────────────────────────────────────────
  # 大腿碰到地形意味着机器人跌倒或严重蹲伏，用于触发终止条件。
  # history_length=4: 保存 4 步历史，避免短暂台阶接触误触发终止。
  thigh_ground_cfg = ContactSensorCfg(
    name="thigh_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=thigh_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  # 每条腿有 2 个小腿（calf）碰撞几何体，共 8 个
  calf_geom_names = tuple(
    f"{leg}_calf_collision{i}" for leg in foot_names for i in (1, 2)
  )

  # ── 接触传感器：小腿触地检测 ─────────────────────────────────────────
  # 小腿（shank）碰地表明腿部姿态异常（过度弯曲），施加负奖励惩罚。
  shank_ground_cfg = ContactSensorCfg(
    name="shank_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=calf_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  # ── 接触传感器：躯干/头部触地检测 ───────────────────────────────────
  # 躯干或头部触地说明机器人已倒地，施加负奖励惩罚。
  trunk_head_ground_cfg = ContactSensorCfg(
    name="trunk_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=("trunk_collision", "head_collision"),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  # 将上面定义的 Go1 专属传感器追加到场景传感器列表中
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    thigh_ground_cfg,
    shank_ground_cfg,
    trunk_head_ground_cfg,
  )

  # 启用地形 curriculum 模式：地形网格按列分配地形类型，按行递增难度。
  # 训练时 terrain_levels_vel 课程函数会根据机器人表现将其移到更难/更容易的行。
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  # GO1_ACTION_SCALE 是关节位置动作的缩放系数，将网络输出
  # 映射到实际关节角度增量（单位弧度）。Go1 使用专属缩放值。
  joint_pos_action.scale = GO1_ACTION_SCALE

  # Viser/Native viewer 的默认视角设置
  cfg.viewer.body_name = "trunk"  # 摄像头跟随的 body
  cfg.viewer.distance = 1.5  # 摄像头与目标 body 的距离，单位米
  cfg.viewer.elevation = -10.0  # 摄像头俯仰角（负值表示稍微俯视）

  # ── 摩擦系数随机化（替换模板中的通用 foot_friction 事件） ──────────────
  # Go1 使用 condim=6（6 自由度摩擦），需要分三轴分别指定：
  #   axes=[0]  -> 滑动摩擦（tangential sliding）
  #   axes=[1]  -> 自旋摩擦（spinning about contact normal）
  #   axes=[2]  -> 滚动摩擦（rolling about tangent axis）
  del cfg.events["foot_friction"]  # 移除模板中仅适用于 condim=3 的单轴版本
  # 滑动摩擦：均匀分布在 [0.3, 1.5]，abs 操作意为直接赋值（非叠加偏移）。
  # shared_random=True 表示同一 episode 内所有足端共享同一随机值，
  # 保证四条腿在同质地面上的一致性。
  cfg.events["foot_friction_slide"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",  # 直接设置绝对值，而非在原值上加减
      "axes": [0],  # 第 0 轴 = 滑动摩擦
      "ranges": (0.3, 1.5),  # 采样区间（无量纲摩擦系数）
      "shared_random": True,  # 四足共享一个随机采样值
    },
  )
  # 自旋摩擦：对数均匀分布在 [1e-4, 2e-2]，对数分布使低值区域有更多样本。
  # 值极小是因为自旋摩擦在正常步态中几乎可忽略，但需给出非零值。
  cfg.events["foot_friction_spin"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",
      "distribution": "log_uniform",  # 对数均匀分布，适合跨越多个数量级的参数
      "axes": [1],  # 第 1 轴 = 自旋摩擦
      "ranges": (1e-4, 2e-2),
      "shared_random": True,
    },
  )
  # 滚动摩擦：对数均匀分布在 [1e-5, 5e-3]，数量级最小（比自旋更弱）。
  cfg.events["foot_friction_roll"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [2],  # 第 2 轴 = 滚动摩擦
      "ranges": (1e-5, 5e-3),
      "shared_random": True,
    },
  )
  # 确保质心随机化作用于 Go1 的 trunk（而非模板默认的其他 body 名称）
  cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)

  # ── 姿态奖励（pose reward）的关节容忍度标准差 ──────────────────────────
  # pose 奖励通过高斯核衡量关节角度偏离参考姿态的程度：
  #   reward = exp(-error^2 / std^2)
  # std 越大表示对该关节偏差越宽容，训练越容忍不同姿态。
  cfg.rewards["pose"].params["std_standing"] = {
    # 静立时：髋关节和大腿关节允许 ±0.05 rad 的小偏差（姿态严格）
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
    # 静立时：小腿关节允许 ±0.1 rad 的偏差（小腿承重，稍宽松）
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    # 步行时：髋/大腿关节允许 ±0.3 rad（步态摆动需要较大自由度）
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
    # 步行时：小腿关节允许 ±0.6 rad（着地阶段弯曲幅度较大）
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # 跑步时：与步行相同的宽松度（跑步不要求严格回到参考姿态）
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
  }

  # 将 upright（直立）奖励绑定到 Go1 的 trunk body，
  # 并使用 terrain_scan 射线传感器估算地形坡度（从而修正"直立"基准方向）
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk",)
  cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
  # body_ang_vel 奖励惩罚躯干角速度，绑定到 trunk body
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)

  # 足部相关奖励绑定到 Go1 各足端的 site 名称
  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # 以下奖励项对 Go1 粗糙地形任务暂时关闭（weight=0 保留接口但不计入总奖励）
  cfg.rewards["body_ang_vel"].weight = 0.0  # 躯干角速度惩罚（由其他项覆盖）
  cfg.rewards["angular_momentum"].weight = 0.0  # 角动量惩罚（粗糙地形步态允许晃动）
  cfg.rewards["air_time"].weight = 0.0  # 腾空时间奖励（暂不鼓励跳跃）

  # ── 碰撞惩罚奖励项 ──────────────────────────────────────────────────
  # 自碰撞（四肢互相接触）：惩罚系数 -0.1，鼓励机器人保持肢体分开
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name},
  )
  # 小腿触地：惩罚系数 -0.1，防止腿部过度弯曲导致小腿着地
  cfg.rewards["shank_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": shank_ground_cfg.name},
  )
  # 躯干/头部触地：惩罚系数 -0.1，防止机器人俯冲或翻滚
  cfg.rewards["trunk_head_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": trunk_head_ground_cfg.name},
  )

  # 粗糙地形上机器人会大幅倾斜，不能仅靠姿态判断是否跌倒；
  # 移除 fell_over 终止条件，改由 out_of_terrain_bounds 来触发重置。
  cfg.terminations.pop("fell_over", None)

  # 大腿触地视为非法接触，触发 episode 终止（说明机器人已严重摔倒）
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": thigh_ground_cfg.name},
  )

  # ── Play 模式覆盖 ──────────────────────────────────────────────────
  if play:
    # 将 episode 时长设为近无限，允许用户长时间观察机器人行为
    cfg.episode_length_s = int(1e9)

    # 关闭观测噪声，使可视化时动作更流畅
    cfg.observations["actor"].enable_corruption = False
    # 移除推力干扰事件，play 模式不随机推倒机器人
    cfg.events.pop("push_robot", None)
    # 移除出界终止条件，play 模式允许机器人自由漫游
    cfg.terminations.pop("out_of_terrain_bounds", None)
    # 清空课程调度，play 模式不进行训练进度管理
    cfg.curriculum = {}
    # 添加随机地形 reset 事件，每次 episode 重置时随机分配地形格子
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        # 保持 curriculum=True 以生成多级难度地形，供 play 模式探索
        cfg.scene.terrain.terrain_generator.curriculum = True
        # 增大列数（地形类型数量），生成更丰富的环境多样性
        cfg.scene.terrain.terrain_generator.num_cols = 500
        # 增大行数（难度级别数量），覆盖从最简单到最难的全范围
        cfg.scene.terrain.terrain_generator.num_rows = 200
        # 加宽边框，避免机器人在地形边缘附近出生时遇到边界问题
        cfg.scene.terrain.terrain_generator.border_width = 100.0

  return cfg


def unitree_go1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 flat terrain velocity configuration."""
  cfg = unitree_go1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensors and collision sensors not needed on flat.
  remove_sensors = {
    "terrain_scan",
    "self_collision",
    "thigh_ground_touch",
    "shank_ground_touch",
    "trunk_ground_touch",
  }
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

  # Remove granular collision rewards (not useful on flat ground).
  for key in ("self_collisions", "shank_collision", "trunk_head_collision"):
    cfg.rewards.pop(key, None)

  # On flat terrain fell_over is sufficient; thigh contact implies fallen.
  cfg.terminations.pop("illegal_contact", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def unitree_go1_stairs_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 stairs velocity configuration."""
  cfg = unitree_go1_rough_env_cfg(play=play)
  site_names = ("FR", "FL", "RR", "RL")

  terrain_size = (8.0, 8.0)
  stair_border_width = 0.5
  stair_platform_width = 2.0
  stair_step_width = 0.4
  min_peak_height = 0.06
  max_peak_height = 0.278
  step_height_range = (
    _step_height_for_peak_height(
      terrain_size,
      stair_border_width,
      stair_platform_width,
      stair_step_width,
      min_peak_height,
    ),
    _step_height_for_peak_height(
      terrain_size,
      stair_border_width,
      stair_platform_width,
      stair_step_width,
      max_peak_height,
    ),
  )

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=terrain_size,
    border_width=20.0,
    num_rows=10,
    num_cols=2,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    sub_terrains={
      "up_stairs": pyramid_stairs_inv(
        proportion=0.5,
        step_height_range=step_height_range,
        step_width=stair_step_width,
        platform_width=stair_platform_width,
        border_width=stair_border_width,
      ),
      "down_stairs": pyramid_stairs(
        proportion=0.5,
        step_height_range=step_height_range,
        step_width=stair_step_width,
        platform_width=stair_platform_width,
        border_width=stair_border_width,
      ),
    },
    add_lights=True,
  )
  cfg.scene.terrain.max_init_terrain_level = 100 if not play else None

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_forward_envs = 1.0
  twist_cmd.resampling_time_range = (8.0, 8.0)
  twist_cmd.ranges.lin_vel_x = (0.4, 0.6)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = None

  cfg.curriculum.pop("command_vel", None)

  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=mdp.track_linear_velocity_dynamic,
    weight=2.0,
    params={
      "command_name": "twist",
      "std": math.sqrt(0.25),
      "max_std": math.sqrt(0.5),
      "min_command": 0.2,
      "max_command": 0.6,
    },
  )
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=mdp.track_angular_velocity_dynamic,
    weight=2.0,
    params={
      "command_name": "twist",
      "std": math.sqrt(0.5),
      "max_std": math.sqrt(0.8),
      "min_command": 0.05,
      "max_command": 0.4,
    },
  )
  cfg.rewards["correct_base_height"] = RewardTermCfg(
    func=mdp.correct_base_height,
    weight=-2.0,
    params={
      "sensor_name": "terrain_scan",
      "target_height": 0.28,
    },
  )
  cfg.rewards["action_smoothness"] = RewardTermCfg(
    func=envs_mdp.action_acc_l2,
    weight=-0.05,
  )
  cfg.rewards["pose"].weight = 0.2
  cfg.rewards["hip_to_default"] = RewardTermCfg(
    func=mdp.hip_to_default_cost,
    weight=-0.05,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(r".*(FR|FL|RR|RL)_hip_joint.*",),
      ),
    },
  )
  cfg.rewards.pop("foot_clearance", None)
  cfg.rewards["feet_regulation"] = RewardTermCfg(
    func=mdp.feet_regulation,
    weight=-0.05,
    params={
      "height_sensor_name": "foot_height_scan",
      "target_height": 0.12,
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
    },
  )

  return cfg
