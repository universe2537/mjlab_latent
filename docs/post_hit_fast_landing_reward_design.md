# Post-Hit Low-Arc Quality Reward Design

## Goal

Add a small auxiliary reward, `post_hit_low_arc_quality_reward`, on top of the
existing hit, net-crossing, landing, stability, and velocity-direction rewards.
The purpose is to discourage unnecessary high lob trajectories after a valid
return and encourage a lower, more pressuring ball that lands sooner after
crossing the net.

This is not a replacement for task success. The existing sparse rewards for
hit, net crossing, and in-bounds landing remain the primary success structure.

## Physical Meaning

The time from hit to first landing is dominated by vertical motion:

```text
z(t) = z0 + vz0 * t - 0.5 * g * t^2
```

where `z0` is post-hit ball height, `vz0` is post-hit vertical velocity, `g` is
gravity, and `t` is time to landing. Shorter post-hit flight time mainly means
the return did not carry excessive upward velocity or apex height.

This reward is therefore a low-arc quality signal. It is not a landing-depth
reward. Horizontal landing distance is mostly controlled by horizontal velocity,
while this term reads the elapsed time between hit and first landing.

## Trigger Conditions

The reward is gated by existing event logic:

1. A valid racket hit occurs.
2. The returned ball crosses the net.
3. The first post-hit landing is detected.
4. By default, that landing must be in bounds (`require_in_bounds=True`).
5. The score is emitted once per hit/return only.

The ordinary hit/cross task uses `TennisHitTracker`. The continuous rally task
uses `TennisContinuousBallState`. Both now record:

- `hit_step`
- `first_bounce_after_hit_step`
- `time_to_landing`
- `first_bounce_after_hit_edge`
- `has_rewarded_fast_landing`

## Mathematical Form

The raw score is bounded and linear:

```python
score = clamp((t_max - time_to_landing) / (t_max - t_min), 0.0, 1.0)
```

Defaults:

```text
fast_landing_t_min = 0.35
fast_landing_t_max = 1.20
fast_landing_weight = 0.05
```

`t_min=0.35` treats very quick legal returns as saturated quality. `t_max=1.20`
stops rewarding slow, high-arc returns. The default weight is intentionally tiny
relative to existing cross and landing rewards (`500`/`1000` in Cross and
larger in LAB variants), so it cannot dominate hit success or stability.

Set `cfg.rewards["post_hit_low_arc_quality_reward"].weight = 0.0` to disable
the term without changing code.

## Relationship To Existing Rewards

Existing terms already handle:

- hit event success;
- net crossing;
- in-bounds landing;
- post-hit x progress;
- post-hit velocity direction;
- lateral-speed suppression;
- robot posture, torque, acceleration, and action-rate regularization.

The new term fills only the missing low-arc/fast-landing quality signal. It
does not alter observation space, action space, termination semantics, or the
meaning of existing rewards.

## Failure Modes Avoided

- It does not reward balls before a valid hit.
- It does not reward balls before net crossing.
- It does not reward hanging the ball into the net.
- With `require_in_bounds=True`, it does not reward out-of-bounds first
  landings.
- It emits once per return through `has_rewarded_fast_landing`.
- It is clamped to `[0, 1]` and has a conservative weight.

## Debug Metrics

The config adds these metrics:

- `first_bounce_after_hit_count`
- `fast_landing_reward_mean`
- `time_to_landing_mean`
- `time_to_landing_min`
- `time_to_landing_max`
- `time_to_landing_valid_count`

Existing metrics already cover:

- `racket_hit_count`
- `crossed_net_count`
- `landing_in_bounds_count`
- `successful_return_count`

## Validation Plan

Use lightweight checks only:

```sh
uv run pytest tests/test_tennis_hit_state.py
uv run pytest tests/test_tennis_task.py
```

For short interactive rollouts, inspect that:

- `crossed_net_count` and `landing_in_bounds_count` remain healthy;
- `time_to_landing_valid_count` is nonzero only after successful returns;
- `time_to_landing_mean` decreases without increasing net faults or out faults;
- `fast_landing_reward_mean` stays finite and within `[0, 1]`;
- robot fall/pose penalties do not worsen.

If training worsens, first set the reward weight to `0.0`, compare success and
fault metrics, then try a smaller weight or wider `[t_min, t_max]` interval.
