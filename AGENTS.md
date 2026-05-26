# Development Workflow

Always use `uv run`, not bare `python`.

Maintain `summary.html` whenever the repository gains useful operational
knowledge, a new task family, a non-obvious debugging lesson, or a training
run worth remembering. Keep it concise and practical: record what changed,
which commands/checkpoints/logs matter, what failed, and the current next
step. Future AI coding agents in this repository should read and update that
file as shared working memory.

```sh
# Type check.
uv run ty check
uv run pyright

# Run tests.
uv run pytest tests/
uv run pytest tests/<test_file>.py

# Format and lint.
uv run ruff format
uv run ruff check --fix
```

The Makefile wraps the common checks:

```sh
make format
make type
make check
make test-fast
make test
make docs
```

Run `make check` before committing. Run `make test` before opening a PR.
When making user-facing changes, add an entry to
`docs/source/changelog.rst` under "Upcoming version (not yet released)"
using Added/Changed/Fixed categories.

For training resumes, runner configs support `load_checkpoint_file`, an
explicit checkpoint path that is resolved independently of `experiment_name`.
`train.py` resolves resume checkpoints in this order:

1. `wandb_run_path`;
2. `agent.load_checkpoint_file`;
3. `agent.load_run` / `agent.load_checkpoint` under the current experiment.

Use `load_checkpoint_file` when a new experiment should continue from a
checkpoint stored under another experiment directory.

# Style

- Line length is 88 columns for code, comments, and docstrings.
- Use 2-space indentation, matching the existing Ruff config.
- Avoid local imports unless they are needed for circular imports or optional
  dependencies.
- Prefer `rg` / `rg --files` for code search.
- Tests should use functions and fixtures, not test classes.
- Keep tests targeted and efficient; prefer running the smallest relevant test
  file during iteration.
- Do not hard-wrap PR or commit-message prose at 88 columns.

# Commits and PRs

- Put `Fixes #<number>` at the end of the commit message body, not in the
  title.
- PR bodies should be plain, concise prose. Do not use section headers,
  checklists, or structured templates.

# Task System Overview

Tasks live under `src/mjlab/tasks`. Importing `mjlab.tasks` auto-imports task
packages, and each robot-specific config module registers task IDs through
`register_mjlab_task`. `load_env_cfg()` and `load_rl_cfg()` return deep copies,
so mutation of a loaded config is expected and does not alter the registry.

Most tasks follow this layout:

- `<task>/<task>_env_cfg.py` defines the robot-agnostic base environment.
- `<task>/config/<robot>/env_cfgs.py` fills in robot assets, body names,
  sensor names, motion files, action scaling, and play-mode overrides.
- `<task>/config/<robot>/rl_cfg.py` defines runner and algorithm config.
- `<task>/config/<robot>/__init__.py` registers task IDs.
- `<task>/mdp/` holds task-specific observations, rewards, terminations,
  commands, actions, events, metrics, and curricula.
- `<task>/rl/` holds custom runners, model wrappers, and export code.

When adding or modifying a task, keep the generic base config robot-agnostic.
Put robot-specific names, regexes, action scales, motion artifacts, and viewer
body names in the robot config.

# Manager-Based RL Conventions

Environment configs are assembled from manager dictionaries:

- `observations`: usually `actor` and `critic`; actor can be noisy, critic is
  normally privileged and clean.
- `actions`: action terms own scaling, offsets, and low-level actuation.
- `commands`: sampled or replayed task targets.
- `events`: reset/startup/interval perturbations and domain randomization.
- `rewards`: scalar terms with weights.
- `terminations`: failure and timeout terms. Use `time_out=True` only for
  truncations.
- `curriculum` and `metrics`: optional task progress and diagnostics.

Contact sensors often run at physics-substep frequency while rewards and
terminations are evaluated once per control step. For short contact events,
configure `history_length=decimation` and read `force_history` instead of only
the instantaneous `force`. Otherwise fast impacts can be visible in simulation
but missed by reward or termination logic.

# Tracking Task

Tracking is a BeyondMimic-style motion imitation task.

- Base config: `src/mjlab/tasks/tracking/tracking_env_cfg.py`.
- G1 config: `src/mjlab/tasks/tracking/config/g1/env_cfgs.py`.
- Core command: `MotionCommand` in `tracking/mdp/commands.py`.
- Runner: `MotionTrackingOnPolicyRunner`.

`MotionCommand` loads one or more `.npz` motion files, keeps per-env
`motion_ids` and local `time_steps`, and exposes reference joint/body state.
Rewards compare the current robot against anchor-aligned reference bodies.
Terminations focus on anchor height/orientation and selected body height error,
so the task tolerates global xy/yaw drift while still catching falls.

The G1 tracking config supplies:

- the G1-with-racket robot asset;
- the motion artifact paths;
- `anchor_body_name="torso_link"`;
- the ordered `body_names` expected by the motion NPZ tensors;
- a self-collision contact sensor using `history_length=4`;
- action scaling from `G1_W_RACKET_ACTION_SCALE`.

Do not reorder `motion_cmd.body_names` unless the motion file tensor layout is
updated at the same time. The order is part of the data contract.

Play mode for tracking disables actor corruption, removes push perturbations,
disables random state initialization, samples motions from the start, and sets
a very long episode length. It does not represent the exact training
distribution.

# Distillation Task

Distillation reuses the tracking environment and adds only a stronger right
wrist encoder-bias event for LATENT-style robustness.

- Base config: `src/mjlab/tasks/distillation/distill_env_cfg.py`.
- G1 config: `src/mjlab/tasks/distillation/config/g1/env_cfgs.py`.
- Runner: `OnlineDistillationRunner`.
- Model: `LatentStudentModel`.
- Observation slicing: `ObservationSlicer`.

The runner implements online DAgger-style latent distillation:

1. Load a frozen tracking teacher via `teacher_task_id` and
   `teacher_checkpoint`.
2. Roll out the student prior policy, optionally mixing teacher actions by
   `teacher_action_prob`.
3. Store `(actor_obs, teacher_action)` in `ReplayBuffer`.
4. Train a conditional VAE with action MSE plus `KL(posterior || prior)`.

`state_terms` are deployment-visible inputs for the prior and decoder.
`target_terms` are training-only inputs for the posterior. Be careful when
changing observation terms in tracking: distillation configs and frozen tennis
decoder checkpoints may depend on exact term names and dimensions.

The student model has:

- posterior `E(z | state, target)`;
- state-conditioned prior `P(z | state)`;
- decoder `D(action | state, z)` unless `z_all=True`.

Tennis uses the distillation checkpoint as a frozen low-level decoder, so
changes to distillation model topology, latent dimension, state terms, or
action dimension can break tennis unless checkpoint metadata is updated and
validated.

# Tennis Tasks

Tennis is a family of high-level latent-control tasks. The robot policy emits
a latent vector; a frozen distillation decoder maps that latent and the current
decoder state terms into low-level G1 joint-position commands.

- Base config: `src/mjlab/tasks/tennis/tennis_env_cfg.py`.
- G1 config: `src/mjlab/tasks/tennis/config/g1/env_cfgs.py`.
- G1 registration: `src/mjlab/tasks/tennis/config/g1/__init__.py`.
- G1 PPO config: `src/mjlab/tasks/tennis/config/g1/rl_cfg.py`.
- Frozen decoder action: `tennis/mdp/actions.py`.
- Ball feeder: `tennis/mdp/ball_providers.py`.
- Hit tracker: `tennis/mdp/hit_state.py`.
- Runner: `TennisLatentOnPolicyRunner`.

The policy action is a latent vector. `FrozenDecoderLatentJointPositionAction`
slices the actor observation into decoder `state_terms`, feeds the latent and
state into a frozen `LatentStudentModel.decode()`, and applies the decoded
joint-position action through a low-level `JointPositionAction`.

The decoder checkpoint is mandatory for normal tennis training. The G1 config
sets the default checkpoint and action scale. The action term validates latent
dimension, state dimension, and checkpoint state-term metadata when available.

The ball is spawned by `RandomFeeder`: it samples a spawn point and a target
landing region, then solves a ballistic trajectory analytically. The target
region is expanded by curriculum.

Registered G1 tennis tasks currently include:

- `Mjlab-Tennis-Hit-Unitree-G1`: first-contact task. Success is the first
  ball-racket contact edge, and the episode terminates immediately on that
  first valid hit. It is useful for learning interception and racket timing.
- `Mjlab-Tennis-Cross-Unitree-G1`: follow-on task. It reuses the same latent
  action space and frozen decoder, but continues the rollout after contact.
  Success requires hitting the ball across the net and having its first
  landing fall inside the opponent singles court.

Cross defaults to continuing from the Hit policy:

```sh
uv run train Mjlab-Tennis-Cross-Unitree-G1
```

Its runner config sets `experiment_name="g1_tennis_latent_cross"`,
`run_name="tennis_cross_from_hit"`, `resume=True`, and
`load_checkpoint_file=DEFAULT_CROSS_RESUME_CHECKPOINT`. Use
`--agent.resume False` to train Cross from scratch, or override
`--agent.load_checkpoint_file /path/to/model.pt` to resume from another
checkpoint. Cross logs still go to the Cross experiment directory.

Hit detection is driven by a contact sensor matching:

- primary: ball geom `tennis_ball`;
- secondary: robot geom `tennis_racket_collision`.

The racket collision geom is a solid thin cylinder approximating the racket
face, not a string mesh or only the rim. Hits inside the disk and on the rim
can both count as racket contact.

Because ball-racket impact is brief, hit detection must use contact history.
`TennisHitTracker` should prefer `force_history` over instantaneous `force`
when the sensor is configured with `history_length=decimation`. A visible
impact may not trigger `racket_hit_event` if only the final substep force is
read.

`TennisHitTracker` owns the shared contact state used by rewards and
terminations. It tracks racket-hit edges, bounce edges, crossed-net edges,
and Cross-specific `landing_in_bounds_edge`. Cross landing success is only
true when the ball has already hit the racket, crossed from the robot side
(`x > net_x`) to the opponent side (`x <= net_x`), and then bounced inside
`landing_x_limits` / `landing_y_limits`.

Cross reward shaping is intentionally staged:

- keep Hit rewards such as `approach_point`, `racket_towards_ball`, and
  `racket_hit_event`;
- add `post_hit_x_progress` to reward leftward ball progress after contact
  while the ball is still on the robot side;
- add sparse `crossed_net_event`;
- add sparse `landing_in_bounds_event`;
- set curriculum success to `landing_in_bounds_after_hit`.

Do not use `first_racket_hit` as a Cross success term; that would collapse
the task back to Hit and can reward juggling or tapping behaviors.

Play mode for tennis only makes the episode length effectively infinite and
disables actor observation corruption. It does not remove hit terminations
unless the CLI is run with `--no-terminations`.

# Other Tasks

Velocity tasks use command-conditioned locomotion with terrain sensors,
foot-contact sensors, domain randomization, curricula, and robot-specific
terrain/play overrides.

Manipulation tasks use staged object-reaching/lifting rewards, object command
terms, end-effector contact sensors, and robot-specific fingertip/site setup.

Cartpole is a compact reference task with local entity/spec construction,
simple observations, and smooth dm-control-style rewards.

# Practical Debugging Notes

- If visuals show an event but reward/termination does not, first check
  timestep, `decimation`, sensor `history_length`, and whether code reads
  history buffers.
- For contact bugs, verify both sides of `ContactMatch` resolve to the intended
  MuJoCo geom/body/subtree names.
- For tracking and distillation, verify observation term order and dimensions
  before loading checkpoints or exporting ONNX.
- For tennis, distinguish physical collision with the robot from contact with
  `tennis_racket_collision`; ball contact with hand/wrist/body geoms will not
  trigger `racket_hit_event`.
- For Tennis Cross, distinguish the intermediate milestones from success:
  `racket_hit_event` and `crossed_net_event` are rewards, while real success
  is `landing_in_bounds_after_hit` on the first in-bounds opponent-court
  bounce.
- Run narrow checks first, for example
  `uv run ty check src/mjlab/tasks/tennis/mdp/hit_state.py`, then broaden to
  `make check` when the change is ready.
