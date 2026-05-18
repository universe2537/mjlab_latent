"""G1 网球任务的场景辅助函数。

球场布局
------------
当前使用标准单打网球场尺寸：

  * 球场总长   : 23.77 m （每侧：网到底线 11.885 m）
  * 单打宽度   : 8.23 m  （边线位于 y = ±4.115 m）
  * 网中央高   : 0.914 m
  * 网立柱高   : 1.07 m
  * 发球线距网 : 6.40 m  （每侧）

机器人侧（"self"）x ∈ (0, 11.885)，对手侧（"opp"）x ∈ (-11.885, 0)。
球网平面位于 x = 0；y 正方向为右手侧（deuce side）。
"""

from __future__ import annotations

import functools

import mujoco

from mjlab.entity import EntityCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils import spec_config as spec_cfg

# ---------------------------------------------------------------------------
# 球场尺寸预设（scale 相对于标准球场的比例）。
# ---------------------------------------------------------------------------
COURT_SIZE_PRESETS: dict[str, float] = {
  "standard": 1.00,  # 23.77 × 8.23 m  — 标准单打球场
  "half": 0.50,  # 11.9 × 4.1 m   — 半场
  "quarter": 0.25,  # 5.9 × 2.1 m    — 四分之一场
  "mini": 0.15,  # 3.6 × 1.2 m    — 迷你场（约 3-5 步）
  "tiny": 0.08,  # 1.9 × 0.66 m   — 极小场（约 2-3 步）
}


def resolve_court_scale(size: str | float) -> float:
  """将字符串预设名称或浮点 scale 解析为标量。

  参数:
    size: 预设名称（如 ``"mini"``）或直接传入的浮点缩放比例。

  返回:
    float 类型的缩放比例，范围 (0, 1]。
  """
  if isinstance(size, str):
    if size not in COURT_SIZE_PRESETS:
      raise ValueError(
        f"Unknown court size {size!r}. Valid options: {list(COURT_SIZE_PRESETS)}"
      )
    return COURT_SIZE_PRESETS[size]
  return float(size)


# ---------------------------------------------------------------------------
# 球场几何常量（供环境配置/课程使用，对外重新导出）。
# ---------------------------------------------------------------------------

# 单侧尺寸。
COURT_HALF_LENGTH = 11.885  # x 方向单侧范围（网到底线）
COURT_HALF_WIDTH = 4.115  # y 方向范围（单打半宽）
SERVICE_LINE_FROM_NET = 6.40  # 发球线距网距离

# 球网。
NET_CENTER_HEIGHT = 0.914
NET_POST_HEIGHT = 1.07
NET_HALF_WIDTH = 5.485  # regulation doubles-post span for the net
NET_THICKNESS_HALF = 0.012

# 网球。
BALL_RADIUS = 0.0335
BALL_MASS = 0.057
BALL_INIT_POS = (0.0, 0.0, 0.0)
BALL_INIT_LIN_VEL = (0.0, 0.0, 0.0)

# 球场线视觉半宽（视觉用）。
_LINE_HALF_W = 0.025  # 沿球场方向（视觉用）
_LINE_HALF_H = 0.003  # 垂直方向

# 底线的 X 坐标（各侧）。
BASELINE_SELF_X = COURT_HALF_LENGTH  # +11.885
BASELINE_OPP_X = -COURT_HALF_LENGTH  # -11.885


def get_tennis_ball_spec() -> mujoco.MjSpec:
  """返回一个自由飞行的网球规格对象。"""
  spec = mujoco.MjSpec()
  spec.add_material(name="tennis_ball_mat", rgba=(0.85, 1.0, 0.05, 1.0))

  body = spec.worldbody.add_body(name="tennis_ball_body")
  body.add_freejoint(name="tennis_ball_freejoint")
  geom = body.add_geom(
    name="tennis_ball",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(BALL_RADIUS,),
    mass=BALL_MASS,
    condim=3,
  )
  geom.material = "tennis_ball_mat"
  geom.friction[:] = (0.8, 0.005, 0.0001)
  geom.solref[:] = (0.004, 1.0)
  geom.solimp[:3] = (0.9, 0.95, 0.001)

  body.add_site(
    name="tennis_ball_center",
    size=(0.01,),
    rgba=(0.85, 1.0, 0.05, 1.0),
  )
  return spec


def get_tennis_ball_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_tennis_ball_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=BALL_INIT_POS,
      lin_vel=BALL_INIT_LIN_VEL,
    ),
  )


def get_tennis_court_spec(scale: float = 1.0) -> mujoco.MjSpec:
  """构建球场（地面、线条、球网）为固定基底实体。

  球场以世界原点为中心，网平面位于 x = 0。
  两侧对称：机器人侧 x ∈ (0, L)，对手侧 x ∈ (-L, 0)，
  其中 L = COURT_HALF_LENGTH * scale。

  参数:
    scale: 球场缩放比例，1.0 为标准尺寸。使用 :func:`resolve_court_scale`
           将预设名称（如 ``"mini"``）转换为此值。
  """
  cl = COURT_HALF_LENGTH * scale
  cw = COURT_HALF_WIDTH * scale
  sln = SERVICE_LINE_FROM_NET * scale
  baseline_self_x = cl
  baseline_opp_x = -cl
  net_hw = NET_HALF_WIDTH * scale
  spec = mujoco.MjSpec()
  spec.add_material(name="court_green", rgba=(0.13, 0.42, 0.22, 1.0))
  spec.add_material(name="court_line_mat", rgba=(0.95, 0.95, 0.95, 1.0))
  spec.add_material(name="tennis_net_mat", rgba=(0.05, 0.05, 0.05, 0.6))
  spec.add_material(name="tennis_net_band_mat", rgba=(0.95, 0.95, 0.95, 1.0))
  spec.add_material(name="tennis_net_post_mat", rgba=(0.2, 0.2, 0.2, 1.0))
  spec.add_material(name="tennis_target_mat", rgba=(0.2, 0.45, 1.0, 0.30))

  body = spec.worldbody.add_body(name="tennis_court_body")

  def add_box(
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    material: str,
    *,
    collidable: bool = False,
  ) -> None:
    geom = body.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=pos,
      size=size,
    )
    geom.material = material
    geom.contype = 1 if collidable else 0
    geom.conaffinity = 1 if collidable else 0

  # --- 球场地面（仅视觉；实际地面由地形平面提供）-
  add_box(
    "court_visual",
    pos=(0.0, 0.0, 0.002),
    size=(cl, cw, 0.002),
    material="court_green",
  )

  # --- 画线 ------------------------------------------------------
  # 底线（垂直于长轴）。
  add_box(
    "court_baseline_self",
    pos=(baseline_self_x, 0.0, 0.006),
    size=(_LINE_HALF_W, cw, _LINE_HALF_H),
    material="court_line_mat",
  )
  add_box(
    "court_baseline_opp",
    pos=(baseline_opp_x, 0.0, 0.006),
    size=(_LINE_HALF_W, cw, _LINE_HALF_H),
    material="court_line_mat",
  )
  # 边线（平行于长轴）。
  add_box(
    "court_sideline_left",
    pos=(0.0, cw, 0.006),
    size=(cl, _LINE_HALF_W, _LINE_HALF_H),
    material="court_line_mat",
  )
  add_box(
    "court_sideline_right",
    pos=(0.0, -cw, 0.006),
    size=(cl, _LINE_HALF_W, _LINE_HALF_H),
    material="court_line_mat",
  )
  # 发球线。
  add_box(
    "court_service_self",
    pos=(sln, 0.0, 0.006),
    size=(_LINE_HALF_W, cw, _LINE_HALF_H),
    material="court_line_mat",
  )
  add_box(
    "court_service_opp",
    pos=(-sln, 0.0, 0.006),
    size=(_LINE_HALF_W, cw, _LINE_HALF_H),
    material="court_line_mat",
  )
  # 中心发球线（两发球线之间）。
  add_box(
    "court_centre_service",
    pos=(0.0, 0.0, 0.006),
    size=(sln, _LINE_HALF_W, _LINE_HALF_H),
    material="court_line_mat",
  )

  # --- 球网 ---------------------------------------------------------------
  # 可碰撞球网网格（单薄箱体，近似布网体积）。
  add_box(
    "tennis_net_collision",
    pos=(0.0, 0.0, NET_CENTER_HEIGHT * 0.5),
    size=(NET_THICKNESS_HALF, net_hw, NET_CENTER_HEIGHT * 0.5),
    material="tennis_net_mat",
    collidable=True,
  )
  # 白色顶带（视觉用）。
  add_box(
    "tennis_net_top_band",
    pos=(0.0, 0.0, NET_CENTER_HEIGHT),
    size=(NET_THICKNESS_HALF * 1.5, net_hw + 0.02, 0.018),
    material="tennis_net_band_mat",
  )
  # 网柱（视觉圆柱）。
  for tag, y_post in (("left", net_hw), ("right", -net_hw)):
    post = body.add_geom(
      name=f"tennis_net_post_{tag}",
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      pos=(0.0, y_post, NET_POST_HEIGHT * 0.5),
      size=(0.03, NET_POST_HEIGHT * 0.5, 0.0),
    )
    post.material = "tennis_net_post_mat"
    post.contype = 0
    post.conaffinity = 0

  # --- 可选落点提示（仅视觉）------------------------
  target = body.add_geom(
    name="target_landing_region",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    pos=(-sln, 0.0, 0.012),
    size=(max(0.3, 1.2 * scale), 0.004, 0.0),
  )
  target.material = "tennis_target_mat"
  target.contype = 0
  target.conaffinity = 0

  # --- 语义参考位置（供事件检测器/课程使用）-
  def add_marker(name: str, pos: tuple[float, float, float]) -> None:
    body.add_site(name=name, pos=pos, size=(0.025,), rgba=(1.0, 1.0, 1.0, 0.7))

  add_marker("net_top_center", (0.0, 0.0, NET_CENTER_HEIGHT))
  add_marker("baseline_self", (baseline_self_x, 0.0, 0.02))
  add_marker("baseline_opp", (baseline_opp_x, 0.0, 0.02))
  add_marker("sideline_left", (0.0, cw, 0.02))
  add_marker("sideline_right", (0.0, -cw, 0.02))
  add_marker("service_self_deuce", (sln * 0.5, cw * 0.5, 0.02))
  add_marker("service_self_ad", (sln * 0.5, -cw * 0.5, 0.02))
  add_marker("service_opp_deuce", (-sln * 0.5, -cw * 0.5, 0.02))
  add_marker("service_opp_ad", (-sln * 0.5, cw * 0.5, 0.02))
  return spec


def get_tennis_court_cfg(scale: float = 1.0) -> EntityCfg:
  """返回可重置到每个环境原点的固定球场实体。

  参数:
    scale: 球场缩放比例。可通过 :func:`resolve_court_scale` 将
           预设名称（如 ``"mini"``）转换为此值。
  """
  return EntityCfg(spec_fn=functools.partial(get_tennis_court_spec, scale=scale))


def get_tennis_terrain_cfg() -> TerrainEntityCfg:
  """返回与独立 MJCF 场景匹配的绿色棋盘格地形平面。"""
  return TerrainEntityCfg(
    terrain_type="plane",
    textures=(
      spec_cfg.TextureCfg(
        name="tennis_court_texture",
        type="2d",
        builtin="checker",
        mark="edge",
        rgb1=(0.13, 0.42, 0.22),
        rgb2=(0.10, 0.32, 0.18),
        markrgb=(0.9, 0.9, 0.9),
        width=300,
        height=300,
      ),
    ),
    materials=(
      spec_cfg.MaterialCfg(
        name="tennis_court",
        texuniform=True,
        texrepeat=(2.0, 2.0),
        reflectance=0.0,
        texture="tennis_court_texture",
        geom_names_expr=("terrain$",),
      ),
    ),
  )
