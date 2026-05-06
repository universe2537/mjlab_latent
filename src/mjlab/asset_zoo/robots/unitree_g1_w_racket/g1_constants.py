"""Unitree G1 持拍版本常量。"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.spec_config import CollisionCfg

G1_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "unitree_g1_w_racket"
  / "xml"
  / "g1_w_racket.xml"
)
assert G1_XML.exists(), f"XML not found: {G1_XML}"


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(G1_XML))


##
# Actuator config.
##


# 电机规格（来自 Unitree）。
ROTOR_INERTIAS_5020 = (
  0.139e-4,
  0.017e-4,
  0.169e-4,
)
GEARS_5020 = (
  1,
  1 + (46 / 18),
  1 + (56 / 16),
)
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_5020, GEARS_5020
)

ROTOR_INERTIAS_7520_14 = (
  0.489e-4,
  0.098e-4,
  0.533e-4,
)
GEARS_7520_14 = (
  1,
  4.5,
  1 + (48 / 22),
)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ROTOR_INERTIAS_7520_22 = (
  0.489e-4,
  0.109e-4,
  0.738e-4,
)
GEARS_7520_22 = (
  1,
  4.5,
  5,
)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

ROTOR_INERTIAS_4010 = (
  0.068e-4,
  0.0,
  0.0,
)
GEARS_4010 = (
  1,
  5,
  5,
)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_4010, GEARS_4010
)

ACTUATOR_5020 = ElectricActuator(
  reflected_inertia=ARMATURE_5020,
  velocity_limit=37.0,
  effort_limit=25.0,
)
ACTUATOR_7520_14 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14,
  velocity_limit=32.0,
  effort_limit=88.0,
)
ACTUATOR_7520_22 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_22,
  velocity_limit=20.0,
  effort_limit=139.0,
)
ACTUATOR_4010 = ElectricActuator(
  reflected_inertia=ARMATURE_4010,
  velocity_limit=22.0,
  effort_limit=5.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

G1_ACTUATOR_5020 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)
G1_ACTUATOR_7520_14 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
G1_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)
G1_ACTUATOR_4010 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  stiffness=STIFFNESS_4010,
  damping=DAMPING_4010,
  effort_limit=ACTUATOR_4010.effort_limit,
  armature=ACTUATOR_4010.reflected_inertia,
)


G1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)
G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.783675),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 1.28,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.76),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# 迁移自 Latent 的 MJCF 与 mjlab 原生 G1 XML使用不同的碰撞 geom 命名。
# 这个持拍版本现在与原生 G1 对齐：
#   - 原生 G1 脚部：left_foot1_collision ... left_foot7_collision
#   - 持拍 G1 脚部：left_foot1_collision ... left_foot7_collision
#   - 部分身体碰撞 geom 没有 ``_collision`` 后缀：thigh、shin、torso
#
# 下面的模式把这些 geom 名称映射到 mjlab 的碰撞策略：
# 身体/自碰撞使用 condim=1；脚-地接触使用 condim=3，并设置 friction 和
# priority，以获得更稳定的地形接触。
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(
    ".*_collision",
    r"^(left|right)_(?:thigh|shin|foot(?:[1-7]_collision)?)$",
    "torso",
  ),
  condim={r"^(left|right)_foot(?:[1-7]_collision)?$": 3, ".*": 1},
  priority={r"^(left|right)_foot(?:[1-7]_collision)?$": 1},
  friction={r"^(left|right)_foot(?:[1-7]_collision)?$": (0.6,)},
)

# 与 FULL_COLLISION 使用同一组 geom，但关闭自碰撞。这个配置与原生 G1 的 helper
# 保持一致，适用于只需要“机器人-环境”接触的任务。
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(
    ".*_collision",
    r"^(left|right)_(?:thigh|shin|foot(?:[1-7]_collision)?)$",
    "torso",
  ),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot(?:[1-7]_collision)?$": 3, ".*": 1},
  priority={r"^(left|right)_foot(?:[1-7]_collision)?$": 1},
  friction={r"^(left|right)_foot(?:[1-7]_collision)?$": (0.6,)},
)

# 更新 feet-only 模式，使其适配原生 G1 形式的 foot1-7 geom 名称。
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot(?:[1-7]_collision)?$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_w_racket_robot_cfg() -> EntityCfg:
  """获取一个新的持拍 G1 机器人配置实例。

  每次都返回新的 EntityCfg，避免同一个配置实例在多个位置共享时被意外修改。
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    # CollisionCfg 会在 MJCF 加载后修改 spec；这里把 Latent 风格 geom 名称
    # 归一化到 mjlab 的接触配置假设。
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_ARTICULATION,
  )


G1_W_RACKET_ACTION_SCALE: dict[str, float] = {}
for a in G1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    G1_W_RACKET_ACTION_SCALE[n] = 0.25 * e / s


# 兼容别名：如果有代码直接导入本模块，并期望使用与
# ``unitree_g1.g1_constants`` 相同的名字，可以继续工作。包级导出仍使用更明确的
# ``get_g1_w_racket_robot_cfg`` / ``G1_W_RACKET_ACTION_SCALE``，避免歧义。
get_g1_robot_cfg = get_g1_w_racket_robot_cfg
G1_ACTION_SCALE = G1_W_RACKET_ACTION_SCALE


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_w_racket_robot_cfg())

  viewer.launch(robot.spec.compile())
