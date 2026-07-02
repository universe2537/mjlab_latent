# AGENTS.md

本文件是给未来 AI Agent / Coding Agent 使用的项目协作规范。

它不是 README 的替代品，而是 Agent 在本仓库工作的行为协议。任何
Agent 在 `mjlab_latent` 中工作时，都必须以当前源码、当前配置和实际运行
产物为准，谨慎保护用户实验资产。

---

## 目录

1. 基本原则
2. 工作区保护规则
3. 信息来源优先级
4. 搜索与阅读规则
5. 常用命令与验证入口
6. Git、changelog 与提交规则
7. 项目目录职责
8. 任务系统总览
9. Manager-Based RL 约定
10. 训练、播放、评测与恢复
11. Tracking 任务
12. Distillation 任务
13. Tennis 任务
14. Velocity / Manipulation / Cartpole
15. 实验可复现规则
16. Vibe Coding 与项目记忆文件
17. 指标分析规则
18. 长任务与 GPU 规则
19. 常见风险清单
20. Agent 开始任务前检查清单
21. Agent 完成任务前检查清单
22. 推荐工作流

---

## 1. 基本原则

### 1.1 Think Before Coding

写代码前必须先理解上下文。执行任何代码修改前，至少完成：

- 阅读用户需求，确认本次目标是分析、运维、代码修改还是实验整理。
- 找到相关源码文件、配置文件、测试文件和运行入口。
- 判断是否涉及 GPU、checkpoint、motion artifact、日志或视频产物。
- 判断是否需要更新 `summary.html`、`AGENTS.md`、`docs/source/changelog.rst`
  或 `VIBE_CODING_LOG.md`。

禁止在没有理解调用路径和配置来源的情况下直接修改文件。

### 1.2 Simplicity First

优先选择最简单、最少侵入、最贴近现有模式的方案。默认不做：

- 大规模重构。
- 新增复杂抽象。
- 引入新依赖。
- 改动无关模块。
- 顺手格式化或清理与当前目标无关的文件。

只有简单方案无法满足目标，或现有结构明显阻碍实现时，才允许引入更复杂
的设计。

### 1.3 Surgical Changes

所有改动必须是外科手术式的：

- 修改范围要小，和用户目标直接相关。
- 不重命名无关变量，不移动无关目录。
- 不删除历史实验、日志、数据、checkpoint、视频或渲染结果。
- 不覆盖用户未提交的本地修改。

如果发现工作区已有未提交修改，必须假设它们是用户的工作。

### 1.4 Use The Project Toolchain

本仓库使用 `uv`。运行 Python 入口时必须使用 `uv run`，不要裸跑
`python`。例如：

```sh
uv run train Mjlab-Tennis-Hit-Unitree-G1
uv run play Mjlab-Tennis-Cross-Unitree-G1 --checkpoint-file <model.pt>
uv run pytest tests/test_tennis_task.py -q
```

### 1.5 Solve The Real Problem

当命令、测试、训练、评测、渲染、下载、GPU/EGL 或数据加载失败时，Agent
必须优先定位和解决根因，不得用更弱的检查、跳过步骤、改小目标或换指标来
假装完成。

- 失败后先保留原始命令、错误、相关配置和日志，再阅读调用路径形成假设。
- workaround 只有在不改变用户目标和数据契约时才可使用；若会改变范围，
  必须明确说明并征求用户确认。
- 若问题来自权限、sandbox、网络、缺失资产、GPU/EGL 不可见或外部服务，
  必须说明证据、请求所需授权或给出具体阻塞点。
- 任务只有在原始目标被验证通过，或 blocker 被清楚记录并给出下一步时，
  才算完成。

---

## 2. 工作区保护规则

### 2.1 禁止破坏用户工作区

禁止执行以下操作，除非用户明确要求并确认风险：

- `git reset --hard`
- `git checkout -- <file>`
- `git clean`
- 删除 `logs/`、`wandb/`、`artifacts/`、`ckpt/`、`data/`
- 删除 checkpoint、视频、TensorBoard event、W&B cache、motion NPZ
- 删除用户手动创建的脚本、实验记录或命令笔记
- 大规模移动目录结构

当前仓库常见本地产物包括 `logs/`、`wandb/`、`artifacts/`、
`ckpt/GEAR-SONIC`、`command.md`、`experiment.html`。它们很可能保存了
实验上下文，不能随手清理。

### 2.2 处理已有修改

如果发现文件已经被修改：

- 不要默认认为是自己造成的。
- 不要自动恢复。
- 如果和当前任务无关，忽略它。
- 如果和当前任务相关，先阅读并理解差异。
- 如果无法判断如何合并，向用户说明冲突点。

### 2.3 只提交自己的改动

当用户要求 commit 时：

- 只 stage 当前任务相关文件。
- 不使用 `git add .`。
- 不把日志、缓存、临时输出、大文件混入 commit。
- commit message 简洁说明本次改动范围。

---

## 3. 信息来源优先级

理解项目行为时，按以下优先级取证：

1. 当前用户需求。
2. 当前仓库源码。
3. 当前仓库配置和 resolved run config。
4. 当前测试。
5. 当前运行日志、TensorBoard/W&B 指标、实验输出。
6. `summary.html` 和 `experiment.html` 的项目记忆。
7. README、docs、changelog。
8. Agent 自己的记忆。

如果源码和文档冲突，以源码和实际配置为准。如果配置默认值和某个 run 的
resolved config 冲突，以该 run 的 resolved config 为准。

注意：`README.md` 是上游项目说明与本地内容混合的文件，尾部可能出现历史
对话或临时需求文本。不要只凭 README 末尾文本判断当前任务要求。

---

## 4. 搜索与阅读规则

优先使用：

```sh
rg <pattern>
rg --files
```

避免低效、噪声大的全仓库 `grep`。修改代码前必须阅读相关文件实际内容，
不能只根据文件名猜测功能。

当用户要求“全量理解项目”时：

- 可以审计源码、配置、脚本、文档、文本日志。
- 不要声称逐行阅读了二进制 checkpoint、视频、缓存文件。
- 对 checkpoint、视频、W&B cache、生成产物只能做清单盘点和元信息记录。

---

## 5. 常用命令与验证入口

### 5.1 开发检查

```sh
# Type check
uv run ty check
uv run pyright

# Tests
uv run pytest tests/
uv run pytest tests/<test_file>.py

# Format and lint
uv run ruff format
uv run ruff check --fix
```

Makefile 包装了常用检查：

```sh
make format
make type
make check
make test-fast
make test
make docs
```

提交前运行 `make check`。开 PR 前运行 `make test`，除非用户明确接受较小
验证范围。

### 5.2 常用窄测试

- 任务注册和通用配置：`uv run pytest tests/test_task_configs.py -q`
- Tracking：`uv run pytest tests/test_tracking_task.py -q`
- Tennis 配置：`uv run pytest tests/test_tennis_task.py -q`
- Tennis hit/cross 状态：`uv run pytest tests/test_tennis_hit_state.py -q`
- Resume 逻辑：`uv run pytest tests/test_train_resume.py -q`
- 单文件类型检查示例：
  `uv run ty check src/mjlab/tasks/tennis/mdp/hit_state.py`

### 5.3 CPU / GPU 测试

测试默认使用 CUDA（如果可用）。需要强制 CPU 时：

```sh
FORCE_CPU=1 uv run pytest tests/test_tennis_task.py -q
make test-cpu-fast
```

### 5.4 代码风格

- 代码、注释和 docstring 行宽保持 88 列。
- 使用 2 空格缩进，匹配 Ruff 配置。
- 避免 local import，除非用于处理循环依赖或可选依赖。
- 测试使用函数和 fixture，不写 test class。
- 测试要窄而快；迭代时优先跑相关单测文件。
- 不要按 88 列硬折 PR body 或 commit message prose。

---

## 6. Git、changelog 与提交规则

- 需要提交时，先运行 `git status --short` 和相关文件的 `git diff`。
- 只 stage 当前任务文件；不要 stage `logs/`、`wandb/`、`.codex/`、
  `.pytest_cache/`、`.ruff_cache/`、临时视频或大 checkpoint。
- commit message body 中的 issue 关闭语句放在末尾，例如 `Fixes #123`，
  不要放 title。
- PR body 保持 plain、concise prose；不要用模板化 section header 或
  checklist，除非用户明确要求。
- 用户可见行为变化需要在 `docs/source/changelog.rst` 的
  `Upcoming version (not yet released)` 下按 `Added` / `Changed` / `Fixed`
  增加条目。

---

## 7. 项目目录职责

| 路径 | 职责 |
| --- | --- |
| `src/mjlab/` | 主包源码。 |
| `src/mjlab/tasks/` | 内置任务族：tracking、distillation、tennis、velocity、manipulation、cartpole。 |
| `src/mjlab/envs/` | Manager-based RL 环境和共享 MDP 工具。 |
| `src/mjlab/managers/` | action、observation、reward、termination、command、event、metric、curriculum 等 manager。 |
| `src/mjlab/rl/` | RSL-RL runner、config、ONNX exporter、vec env wrapper。 |
| `src/mjlab/sensor/` | contact、camera、raycast、builtin 等传感器。 |
| `src/mjlab/asset_zoo/` | 机器人资产、MJCF、机器人常量。 |
| `tests/` | pytest 测试；新增行为应优先添加窄测试。 |
| `docs/` | Sphinx 文档和 changelog。 |
| `scripts/` | 仓库级工具、cloud、benchmark、转换脚本。 |
| `artifacts/` | 本地 motion artifact / 生成资产，不要随手删除。 |
| `logs/rsl_rl/` | 训练、播放、视频、params、event、checkpoint 默认输出根。 |
| `wandb/` | W&B 本地缓存和 run metadata。 |
| `ckpt/` | 外部 checkpoint / ONNX 软链；当前 `ckpt/GEAR-SONIC` 指向 `/data0/universe/ckpt/GEAR-SONIC`。 |
| `summary.html` | 给未来 Agent 的短工作记忆，重要知识变化时更新。 |
| `experiment.html` | 本地实验复盘入口，包含历史 run、指标和视频路径。 |
| `command.md` | 用户本地常用命令笔记；不要未经要求修改或提交。 |

---

## 8. 任务系统总览

任务注册在 `src/mjlab/tasks/registry.py`。

- 导入 `mjlab.tasks` 会自动导入任务包。
- 各 robot-specific config module 通过 `register_mjlab_task()` 注册任务 ID。
- `load_env_cfg()` 和 `load_rl_cfg()` 返回深拷贝；修改加载出的 cfg 不会污染
  registry。
- `load_runner_cls()` 返回任务自定义 runner；没有时使用默认 runner。

典型任务目录结构：

```text
src/mjlab/tasks/<task>/
  <task>_env_cfg.py              # 机器人无关 base env
  config/<robot>/env_cfgs.py     # 机器人资产、body 名、sensor、motion、action scale
  config/<robot>/rl_cfg.py       # runner / PPO / distillation 配置
  config/<robot>/__init__.py     # register_mjlab_task
  mdp/                           # observations/rewards/terminations/actions/commands
  rl/                            # 自定义 runner/model/export
```

新增或修改任务时，保持 base config 机器人无关；把机器人名字、regex、
body ordering、action scale、motion artifact、viewer body 放到 robot config。

当前注册任务族包括：

- `Mjlab-Cartpole-Balance`
- `Mjlab-Cartpole-Swingup`
- `Mjlab-Velocity-Flat-Unitree-G1`
- `Mjlab-Velocity-Rough-Unitree-G1`
- `Mjlab-Velocity-Flat-Unitree-Go1`
- `Mjlab-Velocity-Rough-Unitree-Go1`
- `Mjlab-Velocity-Stairs-Unitree-Go1`
- `Mjlab-Tracking-Flat-Unitree-G1`
- `Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation`
- `Mjlab-Distill-Flat-Unitree-G1`
- `Mjlab-Lift-Cube-Yam`
- `Mjlab-Lift-Cube-Yam-Rgb`
- `Mjlab-Lift-Cube-Yam-Depth`
- `Mjlab-Multi-Cube-Seg-Yam`
- `Mjlab-Tennis-Hit-Unitree-G1`
- `Mjlab-Tennis-Hit-LAB-Unitree-G1`
- `Mjlab-Tennis-Cross-Unitree-G1`
- `Mjlab-Tennis-Cross-LAB-Unitree-G1`
- `Mjlab-Tennis-Cross-Wrist-LAB-Unitree-G1`
- `Mjlab-Tennis-Continuous-Unitree-G1`
- `Mjlab-Tennis-Hit-SONIC-Unitree-G1`
- `Mjlab-Tennis-Hit-SONIC-Encoder-Unitree-G1`
- `Mjlab-Tennis-Cross-SONIC-Unitree-G1`
- `Mjlab-Tennis-Cross-SONIC-Encoder-Unitree-G1`

任务列表可能变化；需要当前列表时运行 `uv run list-envs` 或读取注册文件。

---

## 9. Manager-Based RL 约定

环境配置由 manager 字典组装：

- `observations`：通常包含 `actor` 和 `critic`。actor 可带噪声，
  critic 通常 privileged 且干净。
- `actions`：动作项拥有 action scaling、offset 和底层 actuation。
- `commands`：采样或 replay task targets。
- `events`：reset/startup/interval perturbation 和 domain randomization。
- `rewards`：标量项和权重。
- `terminations`：失败和 timeout。只有 truncation 使用 `time_out=True`。
- `curriculum` / `metrics`：任务进度和诊断。

传感器常以 physics substep 频率更新，而 reward/termination 常在 control step
评估。短促接触事件必须关注 `history_length=decimation` 和
`force_history`；只读 instantaneous `force` 可能漏掉可视化中明显发生的碰撞。

近期 manager 已验证 reward、termination、metrics term 返回 shape 必须是
`(num_envs,)`。新增 term 时不要依赖广播。

---

## 10. 训练、播放、评测与恢复

### 10.1 CLI 与环境变量

入口来自 `pyproject.toml`：

- `train = mjlab.scripts.train:main`
- `play = mjlab.scripts.play:main`
- `demo`
- `list-envs`
- `viz-nan`
- `export-scene`

`train.py` 会读取本地 `.env` 中的简单 `KEY=VALUE` / `export KEY=VALUE`。
常用变量：

- `WANDB_PROJECT`
- `WANDB_ENTITY`（会映射到 `WANDB_USERNAME`）
- `GLI_PATH`
- `MJLAB_MOTION_ALIAS` / `WANDB_ARTIFACT_ALIAS`
- `MJLAB_WARP_QUIET=1`

Tyro 配置使用 Python collection syntax，例如 `--gpu-ids [0,1]`。布尔值遵循
当前 tyro flag 配置，参考现有命令或 `--help`，不要凭记忆改写。

### 10.2 训练输出

默认训练日志：

```text
logs/rsl_rl/<experiment_name>/<run_name_YYYY-MM-DD_HH-MM-SS>/
  params/env.yaml
  params/agent.yaml
  model_*.pt
  events.out.tfevents.*
  videos/train/            # 若 --video True
```

`--log-root` 可改变日志根目录。RSL-RL logger 默认是 W&B，`upload_model=False`
可保留 metric logging 但避免上传模型文件。

### 10.3 GPU 与渲染

`train` 通过 `--gpu-ids` 选择 GPU：

- `None` / 默认通常选择 `[0]`。
- `"all"` 选择当前 `CUDA_VISIBLE_DEVICES` 中所有 GPU。
- 空 `CUDA_VISIBLE_DEVICES` 表示 CPU mode。
- 多 GPU 使用 `torchrunx`，stdout 进入 `TORCHRUNX_LOG_DIR`，默认在 run 目录
  下的 `torchrunx/`。

`MUJOCO_GL=egl` 只在训练视频或 camera sensor 需要渲染时自动设置。历史经验：
普通 sandbox 下 EGL/CUDA 可能不可见；保存视频通常需要在有 GPU/EGL 权限的
环境运行：

```sh
MUJOCO_GL=egl uv run play <TASK> --checkpoint-file <model.pt> \
  --video True --video-length 2000
```

### 10.4 Resume 优先级

当 `agent.resume=True` 时，`train.py` 按以下顺序解析 checkpoint：

1. `wandb_run_path`
2. `agent.load_checkpoint_file`
3. `agent.load_run` / `agent.load_checkpoint` under current experiment

当新 experiment 需要从旧 experiment 的 checkpoint 继续训练时，优先使用
`load_checkpoint_file`，不要误改 `experiment_name` 来找旧 run。

### 10.5 Tracking motion 解析

Tracking 任务有 `MotionCommandCfg.motion_source`：

- `local`：`motion_files` 作为本地 NPZ 路径。相对路径会先尝试
  `${GLI_PATH}/<path>`，再尝试当前工作目录。
- `wandb`：`motion_files` 作为 W&B motion artifact 路径。缺 entity 时使用
  `.env` 中的 `WANDB_ENTITY`。

`play` 使用 trained policy 且只给 `--checkpoint-file` 时，Tracking 任务还需要
`--motion-files`，否则无法从 W&B run 解析 motion artifact。

---

## 11. Tracking 任务

Tracking 是 BeyondMimic-style motion imitation。

- Base config：`src/mjlab/tasks/tracking/tracking_env_cfg.py`
- G1 config：`src/mjlab/tasks/tracking/config/g1/env_cfgs.py`
- Core command：`MotionCommand` in `tracking/mdp/commands.py`
- Runner：`MotionTrackingOnPolicyRunner`

关键约定：

- `MotionCommand` 加载一个或多个 `.npz` motion 文件，维护 per-env
  `motion_ids` 和 `time_steps`，并暴露参考 joint/body state。
- reward 对比当前机器人和 anchor-aligned reference bodies。
- termination 关注 anchor 高度/方向和指定 body 高度误差，允许全局 xy/yaw
  漂移，但会抓倒地。
- G1 tracking 使用 G1-with-racket 资产，默认 motion artifacts 是
  `./artifacts/tennis_random_001` 到 `004` 下的 `motion.npz`。
- `anchor_body_name="torso_link"`。
- `motion_cmd.body_names` 的顺序是 NPZ 张量数据契约，不能随意重排。
- self-collision contact sensor 使用 `history_length=4`。
- action scaling 来自 `G1_W_RACKET_ACTION_SCALE`。
- No-State-Estimation 变体会从 actor obs 移除依赖状态估计的项。

Play mode 会关闭 actor corruption、移除 push、禁用 RSI 随机化、从 motion
开头采样并设置超长 episode。Play mode 不代表训练分布。

---

## 12. Distillation 任务

Distillation 复用 Tracking 环境，只额外增加右手腕 encoder-bias startup event，
用于 LATENT-style robustness。

- Base config：`src/mjlab/tasks/distillation/distill_env_cfg.py`
- G1 config：`src/mjlab/tasks/distillation/config/g1/env_cfgs.py`
- Runner：`OnlineDistillationRunner`
- Model：`LatentStudentModel`
- Observation slicing：`ObservationSlicer`

runner 执行在线 DAgger-style latent distillation：

1. 通过 `teacher_task_id` 和 `teacher_checkpoint` 加载冻结 tracking teacher。
2. rollout student prior policy，可通过 `teacher_action_prob` 混入 teacher action。
3. 将 `(actor_obs, teacher_action)` 存入 `ReplayBuffer`。
4. 训练 conditional VAE：action MSE + `KL(posterior || prior)`。

`state_terms` 是部署可见输入，供 prior 和 decoder 使用。`target_terms` 是训练
专用输入，供 posterior 使用。修改 tracking observation term 名称、顺序或维度
时，必须同步检查 distillation config 和已冻结 tennis decoder checkpoint。

Tennis 默认 decoder checkpoint：

```text
logs/rsl_rl/g1_distillation/distill_cloud_unitree_racket_tennis_2026-05-12_09-35-14/model_30000.pt
```

模型拓扑、latent_dim、state_terms、action_dim 变化都可能破坏 Tennis。

---

## 13. Tennis 任务

Tennis 是高层 latent-control / token-control 任务族。策略输出高层动作；冻结
低层 decoder 将其映射为 G1 joint-position command。

关键文件：

- Base config：`src/mjlab/tasks/tennis/tennis_env_cfg.py`
- G1 config：`src/mjlab/tasks/tennis/config/g1/env_cfgs.py`
- G1 registration：`src/mjlab/tasks/tennis/config/g1/__init__.py`
- G1 PPO config：`src/mjlab/tasks/tennis/config/g1/rl_cfg.py`
- Frozen decoder action：`tennis/mdp/actions.py`
- SONIC action：`tennis/mdp/sonic_actions.py`
- Ball feeder：`tennis/mdp/ball_providers.py`
- Continuous feeder/state：`tennis/mdp/ball_state.py`
- Hit tracker：`tennis/mdp/hit_state.py`
- Runner：`TennisLatentOnPolicyRunner` / `TennisTokenOnPolicyRunner`

### 13.1 Court / coordinate conventions

- `NET_X = 0.0`。
- 机器人在己方半场，常见 reset x 为正。
- Cross 成功需要球从机器人侧 `x > net_x` 越到对手侧 `x <= net_x`，随后在
  opponent singles court 内第一次落地。
- `DEFAULT_COURT_SIZE` 以源码为准；当前是 `"half"`。部分注释可能还写着
  `"mini"`，不要信过期注释。

### 13.2 RandomFeeder

`RandomFeederCfg` 的逻辑是：

1. 在 `spawn_x/y/z_range` 内随机生成球，默认在球网上方附近。
2. 在地面二维 `target_x/y_range` 内采样目标落点，落点 z 固定为 0。
3. 在 `lin_vel_z_range` 内采样初始竖直速度 `vz`。
4. 由 z 方程求飞行时间，再反解 `vx` / `vy`，使球落到目标点。

Hit/Cross 的 `reset_ball` 默认使用同一个 provider；课程会扩展目标落点区域。
修改球生成逻辑时，优先改 provider config 和课程参数，不要把机器人专属区间
写进通用 base config。

### 13.3 Hit / Cross / Continuous

- `Mjlab-Tennis-Hit-Unitree-G1`：first-contact task。成功是第一次有效
  ball-racket contact edge，并立即终止。
- `Mjlab-Tennis-Cross-Unitree-G1`：从 Hit 继续。成功是击球后过网，并在对手
  半场界内第一次落地。不要把 `first_racket_hit` 当成 Cross success。
- `Mjlab-Tennis-Continuous-Unitree-G1`：multi-ball rally。使用
  `TennisContinuousBallState` 和 `OpponentFeederCfg`，对手半场来球、越网清障、
  recovery phase、respawn countdown 和 rally-length curriculum。

Cross 默认 runner config：

```sh
uv run train Mjlab-Tennis-Cross-Unitree-G1
```

它设置：

- `experiment_name="g1_tennis_latent_cross"`
- `run_name="tennis_cross_from_hit"`
- `resume=True`
- `load_checkpoint_file=DEFAULT_CROSS_RESUME_CHECKPOINT`

从头训练 Cross 用：

```sh
uv run train Mjlab-Tennis-Cross-Unitree-G1 --agent.resume False
```

从其他 checkpoint 继续：

```sh
uv run train Mjlab-Tennis-Cross-Unitree-G1 \
  --agent.load-checkpoint-file /path/to/model.pt
```

Continuous 默认从 Cross checkpoint warm-start，但 `reset_resume_progress=True`，
即只加载权重，重置 optimizer / iteration / RND / env progress，避免旧 step
计数跳过 early curriculum。

### 13.4 Contact and success tracking

Hit detection 使用 contact sensor：

- primary：ball geom `tennis_ball`
- secondary：robot geom `tennis_racket_collision`

`tennis_racket_collision` 是近似球拍面的实体薄圆柱，不只是球拍框。球碰到
手、腕、身体几何不会触发 `racket_hit_event`。

由于球拍接触极短，`TennisHitTracker` 和相关逻辑必须优先读
`force_history`。看到视频里碰到了但 reward/termination 没触发时，先检查：

- `decimation`
- contact sensor `history_length`
- `force_history` 是否存在并被读取
- `ContactMatch` 是否解析到正确 MuJoCo geom/body/subtree 名称

Cross reward shaping 是分阶段的：

- 保留 Hit 的 `approach_point`、`racket_towards_ball`、`racket_hit_event`。
- 加 `post_hit_x_progress` 和 `post_hit_ball_velocity_direction`。
- 加 sparse `crossed_net_event`。
- 加 sparse `landing_in_bounds_event`。
- curriculum success 使用 `landing_in_bounds_after_hit`。

### 13.5 LAB / Wrist residual

- Hit-LAB / Cross-LAB 使用 LATENT-style barrier：
  `z = prior_mean + scale * prior_std * tanh(action)`。
- Cross-Wrist-LAB 当前语义是从旧 Cross latent action 迁移到
  `16 latent + 3 wrist residual`，直接微调右腕
  `roll/pitch/yaw`，默认 residual scale `(0.03, 0.05, 0.05)`。
- runner 支持扩展 actor action head，把旧 16 维 policy 迁移到 19 维动作空间。

### 13.6 SONIC

SONIC 任务使用 ONNX deployment decoder / encoder：

- decoder：`ckpt/GEAR-SONIC/model_decoder.onnx`
- encoder：`ckpt/GEAR-SONIC/model_encoder.onnx`
- `ckpt/GEAR-SONIC` 是本地软链，当前指向 `/data0/universe/ckpt/GEAR-SONIC`。

Direct-token 任务输出 64 维 token。Encoder-prior 任务使用 SONIC encoder 作为
token prior，策略输出 bounded residual：

```text
token = encoder(current_proprio_history) + token_residual_scale * tanh(action)
```

历史实验显示 direct-token Hit 可学，但 Cross/encoder-prior 需要谨慎验证。不要
把 SONIC encoder-prior 作为默认主线，除非视频、root stability 和 hit metrics
都支持。

### 13.7 Tennis 常用验证

```sh
uv run pytest tests/test_tennis_task.py tests/test_tennis_hit_state.py -q
uv run ty check src/mjlab/tasks/tennis/mdp/hit_state.py
uv run ty check src/mjlab/tasks/tennis/mdp/ball_providers.py
```

---

## 14. Velocity / Manipulation / Cartpole

- Velocity：command-conditioned locomotion，包含 flat/rough/stairs、terrain
  sensors、foot contact、domain randomization、curriculum 和 robot-specific
  terrain/play overrides。
- Manipulation：Yam cube lifting / RGB / depth / multi-cube segmentation，
  使用 object command、fingertip/site setup、staged reaching/lifting rewards。
- Cartpole：紧凑 reference task，适合作为 task registry、local entity/spec
  construction 和 simple reward 的参考。

---

## 15. 实验可复现规则

每次正式实验必须能回答：

- 使用了哪个代码版本？
- 使用了哪个 task ID？
- 使用了哪个 config / CLI override？
- 使用了哪个 checkpoint？
- 使用了哪些 motion/data/artifact？
- 启动命令是什么？
- 环境变量是什么？
- 输出路径在哪里？
- 指标文件和 TensorBoard event 在哪里？
- 是否有视频或渲染结果？
- 是否能重复运行？

建议每个重要实验目录或实验笔记包含：

```text
COMMAND.md
config.yaml 或 params/env.yaml + params/agent.yaml
SUMMARY.md
metrics.json
stdout.log
videos/
```

不要把 `last checkpoint` 自动等同于 `best checkpoint`。Tennis 历史实验尤其
需要结合视频、`Episode_Metrics/*`、termination 分布和 `Policy/mean_std` 判断。

---

## 16. Vibe Coding 与项目记忆文件

### 16.1 summary.html

必须维护 `summary.html`，当仓库获得以下知识时更新：

- 有用的操作经验。
- 新任务族或任务语义变化。
- 非显然调试教训。
- 值得记住的训练 run、checkpoint、日志或视频。
- 当前 next step 发生变化。

保持简洁实用：写清改变了什么、重要命令/路径/checkpoint/log、失败了什么、
下一步是什么。

### 16.2 experiment.html

`experiment.html` 是本地实验复盘入口，包含历史 run、指标解释、视频路径和
路线判断。它非常有用，但可能过期；读取后仍需用当前源码、当前日志和当前
run config 交叉验证。

### 16.3 VIBE_CODING_LOG.md

重大代码修改任务使用 `VIBE_CODING_LOG.md` 作为过程账本。如果文件不存在，
首次重大任务可以创建。

以下情况必须写入：

- 新增功能。
- 修改核心训练、评测、部署、数据转换流程。
- 修改多个文件且可能影响已有行为。
- 多轮尝试后才解决的问题。
- 用户要求记录过程。

只读分析、路径检查、简单日志查看、启动教程默认不写。

### 16.4 记录边界

不要把所有小改动都写入项目记忆。以下修改默认只改目标文件，不额外更新
`summary.html`、`VIBE_CODING_LOG.md` 或 `docs/source/changelog.rst`：

- 纯文字、注释、命令笔记、协作规范或本地说明的小幅修正。
- 不改变代码行为、训练配置、评测流程、数据契约或实验产物的小改动。
- 用户明确说“不用记录”的整理性修改。

以下情况仍然必须记录：影响训练/评测/部署行为，改变任务语义或 observation /
reward / termination / provider 契约，新增或迁移实验资产，产生重要经验教训，
或用户明确要求记录。若不确定，优先在最终回复中说明判断，而不是机械更新多个
记忆文件。

建议格式：

```md
## YYYY-MM-DD HH:MM - 任务名称

### 目标

### 计划

- [ ] 子任务 1
- [ ] 子任务 2

### 预期改动

### 进展记录

### 问题记录

#### 问题 1：标题

现象：
可能原因：
验证方式：
验证结果：
状态：Blocked / Resolved
最终总结：
```

---

## 17. 指标分析规则

分析训练或评测指标时，不要只看名字。必须确认：

- 指标在哪里写入。
- 是 step 级别还是 episode 级别。
- 是均值、累计值、事件计数还是滑动窗口。
- 是否经过 reward weight 或 time scaling。
- 是否受异常值影响。
- 是否和 TensorBoard 显示尺度一致。
- 是否有 per-case 明细、episode metrics 或 termination breakdown。

Tennis 特别注意：

- `Episode_Reward/*` 是加权 reward，不等于原始事件频次。
- `Episode_Metrics/*` 更适合解释 sparse events。
- Hit 看 `first_racket_hit` / `racket_hit_event`。
- Cross 看 `landing_in_bounds_after_hit`，不要只看 hit。
- `second_contact` 高通常表示挡球/粘球局部最优。
- `root_height` / `bad_orientation` 高说明动作先验或控制输出破坏站立。
- `Policy/mean_std` 数百到上千说明高层策略探索分布非常吵，reward 高也可能
  不稳定。

---

## 18. 长任务与 GPU 规则

适用于训练、评测、渲染、数据批量转换、下载或上传。

长时间训练默认优先使用 `tmux` 挂起，尤其是用户明确要求“启动训练”、
“挂起任务”或“detach”时。推荐形式：

```sh
tmux new-session -d -s <session_name> -c /home/universe/workspace/mjlab_latent \
  '<env vars> uv run train <TASK_ID> <overrides>'
```

启动后必须记录 session 名、task ID、GPU、run name、输出目录或日志查看方式。
停止训练优先使用温和方式，例如 `tmux send-keys -t <session_name> C-c`；不要用
`kill -9`，除非进程无法正常退出且用户确认。若 sandbox 无法访问 host tmux
socket，需要按授权流程申请执行 `tmux`，用户明确允许后可以继续启动。

启动前说明：

- 使用哪个环境。
- 使用哪些 GPU。
- 输入数据和 checkpoint 路径。
- 输出路径。
- 预计产物。
- 如何查看日志。
- 如何停止任务。

运行中关注：

- 进程是否存在。
- GPU 是否占用正确。
- 日志是否增长。
- checkpoint / event / video 是否生成。
- 是否有异常错误或 NaN。

结束后检查：

- 没有残留异常进程。
- 没有占用错误 GPU。
- 输出文件存在并可读。
- 不要杀掉其他用户进程。

---

## 19. 常见风险清单

- 软链指错位置，尤其 `ckpt/GEAR-SONIC`。
- 相对路径在不同 cwd 下失效。
- Tracking 读取的 motion NPZ 不是预期数据。
- Motion NPZ body order 与 `motion_cmd.body_names` 不一致。
- 评测 checkpoint 正在被训练覆盖。
- `last checkpoint` 不一定最好。
- `load_checkpoint_file`、`load_run`、`wandb_run_path` 优先级被误解。
- W&B artifact 缺 entity，依赖 `.env` 中 `WANDB_ENTITY`。
- Tennis 物理碰撞和 `tennis_racket_collision` 接触不是一回事。
- Contact sensor 没有 history，导致短接触被漏检。
- Cross success 被误设为 first hit。
- Continuous recovery phase 中旧球 fault 被误判。
- TensorBoard 曲线被异常值拉歪。
- 多 GPU `torchrunx` 日志位置被忽略。
- 渲染时缺 EGL/CUDA 权限。
- 误删 `logs/`、`wandb/`、`artifacts/`、checkpoint、视频。
- 把用户的 `command.md` 或本地实验复盘误提交。

---

## 20. Agent 开始任务前检查清单

每次任务开始前快速检查：

- 当前用户目标是什么？
- 任务类型是分析、运维、代码修改还是实验整理？
- 是否需要写 `VIBE_CODING_LOG.md`？
- 是否需要更新 `summary.html` 或 changelog？
- 相关源码在哪里？
- 相关配置在哪里？
- 输入数据、motion、checkpoint 在哪里？
- 输出应该在哪里？
- 是否涉及 GPU、W&B、EGL、视频？
- 是否可能影响用户已有工作？
- 完成后如何验证？

---

## 21. Agent 完成任务前检查清单

结束前检查：

- 目标是否真的完成？
- 是否运行了和风险匹配的验证？
- 是否有未说明的失败或跳过的检查？
- 是否有临时文件需要清理？
- 是否有长进程还在跑？
- 是否有输出路径、checkpoint、视频路径需要告诉用户？
- 是否需要更新 `summary.html`、`VIBE_CODING_LOG.md`、changelog？
- 是否需要 commit，且只包含本次任务文件？
- 是否需要提醒后续风险？

---

## 22. 推荐工作流

### 22.1 分析任务

1. 读需求。
2. 用 `rg` 找相关源码、配置、测试、日志。
3. 读取实际内容。
4. 给结论。
5. 给依据。
6. 说明尚未验证的部分。

### 22.2 代码修改任务

1. 读需求和相关源码。
2. 形成短计划。
3. 如属重大修改，写 `VIBE_CODING_LOG.md`。
4. 小步修改。
5. 跑窄测试或类型检查。
6. 按风险扩大到 `make check` / `make test`。
7. 更新项目记忆或 changelog。
8. 如用户要求，提交只包含本任务的 commit。

### 22.3 训练任务

1. 确认 task ID、config、motion/data、checkpoint。
2. 确认 GPU 和日志根目录。
3. 明确启动命令。
4. 启动或交给用户启动。
5. 说明日志、TensorBoard、W&B、checkpoint 和停止方式。

### 22.4 评测 / 播放 / 渲染任务

1. 确认 checkpoint。
2. 确认 motion / test distribution。
3. 确认 viewer 或 video mode。
4. 如需视频，确认 `MUJOCO_GL=egl` 和输出路径。
5. 运行后读取 metrics 或检查视频文件。
6. 总结结果和产物位置。

### 22.5 实验整理任务

1. 只读盘点。
2. 分类目录职责。
3. 识别活跃进程。
4. 标记可删除、可归档、必须保留内容。
5. 给整理方案。
6. 等用户确认后再移动或清理。

---

## 最终目标

本规范的目标不是限制 Agent，而是让 Agent 的行为可预测、可追踪、可复现。
一个合格的 Agent 应该做到：

- 知道自己为什么读这个文件。
- 知道自己为什么改这几行。
- 知道怎么验证这次改动。
- 知道哪些东西不能碰。
- 知道失败后如何留下经验。
- 知道如何把一次任务变成未来可复用的知识。
