# Post-Hit Fast Landing Reward Audit

Date: 2026-06-14

This audit checks whether the tennis task already contains a reward equivalent
to "after a valid hit and successful net crossing, encourage the returned ball
to reach its first landing sooner." The existing code has hit, net-crossing,
landing, velocity-direction, and success metrics, but no direct hit-to-landing
flight-time or low-arc quality reward.

| 检查项 | 是否已有 | 文件路径 | 关键函数/变量 | 说明 |
| --- | --- | --- | --- | --- |
| tennis task 环境文件 | 是 | `src/mjlab/tasks/tennis/tennis_env_cfg.py` | `make_tennis_latent_env_cfg`, `make_tennis_latent_cross_env_cfg`, `make_tennis_continuous_env_cfg` | 配置 scene、observations、rewards、terminations、metrics、curriculum。 |
| reward function 文件 | 是 | `src/mjlab/tasks/tennis/mdp/rewards.py`, `src/mjlab/tasks/tennis/mdp/ball_state.py` | `racket_hit_event`, `post_hit_x_progress`, `post_hit_ball_velocity_direction`, continuous variants | 普通 hit reward 在 `rewards.py`；continuous reward 在 `ball_state.py`。 |
| 已有击球 reward | 是 | `src/mjlab/tasks/tennis/mdp/rewards.py`, `src/mjlab/tasks/tennis/mdp/ball_state.py` | `racket_hit_event`, `continuous_racket_hit_event` | 稀疏奖励首次球拍接触。 |
| 已有过网 reward | 是 | 同上 | `crossed_net_event`, `continuous_crossed_net_event` | 击球后首次过网 edge。continuous 版本还检查过网高度不低于网高。 |
| 已有落地/落点 reward | 是 | 同上 | `landing_in_bounds_event`, `continuous_landing_in_bounds_event` | 球过网后首次落入对方界内给稀疏成功奖励。 |
| 已有球速 reward | 部分 | 同上 | `post_hit_ball_velocity_direction`, `continuous_post_hit_ball_velocity_direction` | 奖励 -x 方向速度并抑制横向速度；不评价竖直飞行时间或高吊球。 |
| 姿态/稳定性 reward | 是 | `src/mjlab/tasks/tennis/tennis_env_cfg.py` | `alive`, `joint_pos_limits`, `joint_torques_l2`, `joint_acc_l2`, `fall_penalty`, recovery pose terms | 已有稳定性和动作正则项。 |
| episode logging / metrics | 是 | `src/mjlab/tasks/tennis/mdp/metrics.py`, `src/mjlab/tasks/tennis/mdp/ball_state.py`, `tennis_env_cfg.py` | `MetricsTermCfg`, `*_count_metric` | 已有 hit/cross/landing/success/fault/recovery metrics。 |
| ball state 表示 | 是 | `src/mjlab/tasks/tennis/mdp/observations.py`, `ball_state.py` | `ball.data.root_link_pos_w`, `ball.data.root_link_lin_vel_w` | 可读取位置和速度。 |
| racket state 表示 | 是 | `observations.py`, `rewards.py` | `tennis_racket_center`, `racket_to_ball_b`, `racket_velocity_b` | 球拍中心 site 和速度用于接近/速度 reward。 |
| ball-racket contact / hit event | 是 | `hit_state.py`, `ball_state.py` | `racket_hit_edge`, `racket_hit_count`, `has_racket_hit` | 由 contact sensor edge 检测。 |
| ball crossing net 检测 | 是 | `hit_state.py`, `ball_state.py` | `crossed_net_edge`, `has_crossed_net`, `_interpolate_net_z` | 普通版本用 x 过网；continuous 版本插值检查过网高度。 |
| ball ground contact / bounce 检测 | 是 | `hit_state.py`, `ball_state.py` | `bounce_edge` | 用 vz 从负到非负且球高度接近地面检测。 |
| reward 权重配置方式 | 是 | `tennis_env_cfg.py`, `config/g1/env_cfgs.py` | `RewardTermCfg(weight=...)` | 权重在环境配置中显式设置，可设为 0 关闭。 |
| 击球后飞行时间 / 低弧线 reward | 否 | 新增前无 | 无 | 未发现 time-to-landing、low trajectory、fast landing 或 apex-height penalty。 |
| 高吊球惩罚 | 否 | 新增前无 | 无 | `z_limits` 只作为 out-of-play/fault 边界，不是质量 shaping。 |
| 避免重复奖励 | 部分已有 | `hit_state.py`, `ball_state.py` | `~has_landed_in_bounds`, `successful_return_edge` | 成功落地 edge 已一次性；新增项复用一次性 edge 并增加 `has_rewarded_fast_landing`。 |

重点回答：

- 当前没有等价的“击球后飞行时间”或“低弧线”奖励。
- 当前能判断 hit event：`racket_hit_edge` / `continuous_racket_hit_event`。
- 当前能判断球过网：`crossed_net_edge` / `continuous_crossed_net_event`。
- 当前能判断击球后第一次落地：已有 `bounce_edge` 和成功落地 edge；本次补充 `first_bounce_after_hit_edge`。
- 新增前不能记录 `hit_time` / `first_bounce_time`；本次用 `hit_step`、`first_bounce_after_hit_step` 和 `env.step_dt` 记录。
- 当前已有一次性落地成功 edge，可避免同一次击球重复成功奖励；本次新增 `has_rewarded_fast_landing` 避免重复质量奖励。
- 当前已有 reward logging 机制：`MetricsTermCfg`，本次复用并补充 time-to-landing metrics。

查漏补缺结论：

- 已有 reward 覆盖了有效击球、过网、界内落地、回球方向速度、球拍接近、球拍朝球速度、姿态稳定和动作正则。
- 已有 reward 没有覆盖击球后竖直飞行时间、低弧线质量、apex height penalty 或高吊球惩罚。
- 新增项应复用已有 hit/cross/landing edge，而不是重写事件系统。
