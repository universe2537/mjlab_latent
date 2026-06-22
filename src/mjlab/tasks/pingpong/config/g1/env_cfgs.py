"""Unitree G1 configuration for table-tennis latent-control tasks."""

from __future__ import annotations

import mujoco

from mjlab.asset_zoo.robots import G1_W_RACKET_ACTION_SCALE
from mjlab.asset_zoo.robots.unitree_g1_w_racket.g1_constants import (
  FULL_COLLISION,
  G1_ARTICULATION,
  KNEES_BENT_KEYFRAME,
)
from mjlab.asset_zoo.robots.unitree_g1_w_racket.g1_constants import (
  get_spec as get_g1_w_racket_spec,
)
from mjlab.entity import EntityCfg
from mjlab.tasks.pingpong.pingpong_env_cfg import (
  make_pingpong_latent_env_cfg,
  make_pingpong_latent_return_env_cfg,
)
from mjlab.tasks.pingpong.scene import get_pingpong_ball_cfg, get_pingpong_table_cfg
from mjlab.tasks.tennis.mdp import FrozenDecoderLatentJointPositionActionCfg

DEFAULT_DECODER_CHECKPOINT = "logs/rsl_rl/g1_distillation/distill_cloud_unitree_racket_tennis_2026-05-12_09-35-14/model_30000.pt"


def _iter_body_tree(body: mujoco.MjsBody):
  yield body
  for child in body.bodies:
    yield from _iter_body_tree(child)


def _find_geom(spec: mujoco.MjSpec, name: str) -> mujoco.MjsGeom:
  for body in _iter_body_tree(spec.worldbody):
    for geom in body.geoms:
      if geom.name == name:
        return geom
  raise ValueError(f"Could not find geom {name!r} in G1 racket spec.")


def _find_site(spec: mujoco.MjSpec, name: str) -> mujoco.MjsSite:
  for body in _iter_body_tree(spec.worldbody):
    for site in body.sites:
      if site.name == name:
        return site
  raise ValueError(f"Could not find site {name!r} in G1 racket spec.")


def get_g1_w_pingpong_paddle_spec() -> mujoco.MjSpec:
  """Reuse the held-racket G1 XML with a smaller pingpong paddle surface."""
  spec = get_g1_w_racket_spec()
  paddle = _find_geom(spec, "tennis_racket_collision")
  paddle.name = "pingpong_paddle_collision"
  paddle.size[0] = 0.075
  paddle.size[1] = 0.004
  paddle.pos[:] = (0.1025, -0.004, 0.26)
  paddle.rgba[:] = (0.85, 0.12, 0.06, 0.35)

  center = _find_site(spec, "tennis_racket_center")
  center.name = "pingpong_paddle_center"
  center.pos[:] = (0.1025, -0.004, 0.26)
  center.size[0] = 0.01
  return spec


def get_g1_w_pingpong_paddle_robot_cfg() -> EntityCfg:
  """Return a G1 robot cfg with a table-tennis paddle collision proxy."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_g1_w_pingpong_paddle_spec,
    articulation=G1_ARTICULATION,
  )


def _apply_g1_pingpong_common(cfg, play: bool):
  cfg.scene.entities = {
    "robot": get_g1_w_pingpong_paddle_robot_cfg(),
    "ball": get_pingpong_ball_cfg(),
    "table": get_pingpong_table_cfg(),
  }
  cfg.viewer.body_name = "torso_link"
  cfg.viewer.elevation = -16.0
  cfg.viewer.azimuth = 135.0

  action = cfg.actions["latent_joint_pos"]
  assert isinstance(action, FrozenDecoderLatentJointPositionActionCfg)
  action.scale = G1_W_RACKET_ACTION_SCALE
  action.decoder_checkpoint = DEFAULT_DECODER_CHECKPOINT

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
  return cfg


def unitree_g1_pingpong_latent_hit_env_cfg(play: bool = False):
  """Create G1 table-tennis legal-hit task."""
  cfg = make_pingpong_latent_env_cfg()
  return _apply_g1_pingpong_common(cfg, play=play)


def unitree_g1_pingpong_latent_return_env_cfg(play: bool = False):
  """Create G1 table-tennis legal-return task."""
  cfg = make_pingpong_latent_return_env_cfg()
  return _apply_g1_pingpong_common(cfg, play=play)


__all__ = [
  "DEFAULT_DECODER_CHECKPOINT",
  "get_g1_w_pingpong_paddle_robot_cfg",
  "get_g1_w_pingpong_paddle_spec",
  "unitree_g1_pingpong_latent_hit_env_cfg",
  "unitree_g1_pingpong_latent_return_env_cfg",
]
