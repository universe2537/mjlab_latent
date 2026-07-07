## 2026-07-06 - Pingpong PACE Baseline Task

### 目标

新增独立 `Mjlab-Pingpong-PACE-Unitree-G1` baseline，用 direct joint-position
action、PACE-style learned ball predictor 和 PACE reward 思路对比现有 latent
Cross / StrikeQuality；不修改现有 Hit、Cross、StrikeQuality、decoder 或
distillation checkpoint。

### 计划

- [x] 新增 PACE env factory 和 G1 task registration。
- [x] 使用 `JointPositionActionCfg` + `G1_W_RACKET_ACTION_SCALE`，移除 frozen
  latent decoder action。
- [x] 新增 table-local PACE observation、prediction state、reward helpers。
- [x] 新增 `PingpongPaceOnPolicyRunner` 和 predictor checkpoint keys。
- [x] 补配置、state/prediction 和 runner config 测试。
- [x] 跑 CPU smoke train 验证真实训练 loop。

### 实现记录

- 新 task ID：`Mjlab-Pingpong-PACE-Unitree-G1`。
- 默认 experiment / run：
  `g1_pingpong_pace` / `pingpong_pace_scratch`。
- runner：`PingpongPaceOnPolicyRunner`，会在 rollout 中记录最近球位历史，
  训练 5 帧 history -> future ball pose 的 MLP predictor，并在 checkpoint
  中额外保存 `pred_state_dict`、`pred_optimizer_state_dict`、`pred_cfg`。
- actor obs shape smoke 为 `105`，critic obs shape 为 `120`；action shape 为
  `29` direct joint-position。
- PACE core reward 权重：
  `pace_contact=150`、`pace_future_ee_target=2`、
  `pace_future_body_target=5`、`pace_future_base_vel_target=5`、
  `pace_future_landing_distance=60`、`pace_future_pass_net=100`、
  `pace_table_success=100`。
- 普通 `Cross` 和 `Cross-StrikeQuality` 未接入 `pace_*` reward；PACE 仅作为
  独立 baseline task。

### 验证记录

- 编译检查通过：
  `UV_CACHE_DIR=/tmp/uv-cache uv run python -m py_compile ...`
- 单测通过：
  `FORCE_CPU=1 UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pingpong_task.py tests/test_pingpong_state.py -q`
  -> `23 passed`。
- 类型检查通过：
  `UV_CACHE_DIR=/tmp/uv-cache uv run ty check src/mjlab/tasks/pingpong tests/test_pingpong_task.py`
  -> `All checks passed!`。
- CPU smoke train 通过：
  `CUDA_VISIBLE_DEVICES= FORCE_CPU=1 WANDB_MODE=offline UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run train Mjlab-Pingpong-PACE-Unitree-G1 --agent.max-iterations 1 --env.scene.num-envs 2 --agent.upload-model False`。
  该 run 写入
  `logs/rsl_rl/g1_pingpong_pace/pingpong_pace_scratch_2026-07-06_20-33-59`，
  第 0 iteration 记录 `predictor_mse ~= 1.0167`。
- 计划中的 `--env.num-envs` 在当前 CLI 下无效，实际参数是
  `--env.scene.num-envs`。

## 2026-07-06 - Pingpong Cross-StrikeQuality PACE-Style Reward Shaping

### 目标

只在 `Mjlab-Pingpong-Cross-StrikeQuality-Unitree-G1` 中接入 PACE-style
击球后预测质量 reward，保留普通 `Mjlab-Pingpong-Cross-Unitree-G1` baseline
不变，本轮不加入 base 站位或速度 reward。

### 计划

- [x] 保守启用已有预测型 reward：过网 clearance、对方台面落点、击球后速度。
- [x] 复用 `PingpongRallyState` 已有预测结果，不新增独立球路预测器。
- [x] 保持 Cross-Impact 和 Cross-StrikeQualityEnergyRelax 继续继承
  Cross-StrikeQuality，只叠加各自原有差异。
- [x] 更新配置测试并运行窄验证。

### 实现记录

- `CROSS_STRIKE_QUALITY_REWARD_WEIGHTS` 现在启用：
  `strike_pred_net_clearance=40.0`，
  `strike_pred_landing_inside=80.0`，
  `strike_post_hit_speed=30.0`。
- `_add_strike_quality_rewards()` 向 StrikeQuality cfg 添加三项 reward term；
  普通 Cross 的 reward 配置保持不变。
- 验证命令：
  `FORCE_CPU=1 UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pingpong_task.py -q`
  和
  `UV_CACHE_DIR=/tmp/uv-cache uv run ty check src/mjlab/tasks/pingpong tests/test_pingpong_task.py`
  均通过。

### 训练环境数与 GPU 选择

- 历史 Pingpong env count：
  - Hit V1：`8192` env，GPU `2,4`。
  - Hit V3/V4 clean/collision：常用 `10240` env（`512*20`），GPU
    `1,2`。
  - Cross stable：`16384` env，GPU `4,6`。
  - Cross-Diag / Cross-StrikeQuality 旧线：`16384` env，GPU `0,1`。
  - Cross-Impact 长训：先试每卡 `10240` env，后改为每卡 `15360`
    env（`512*30`），GPU `4,5,6,7`，总 `61440` env。
- 当前机器状态（2026-07-06 15:07）：GPU `1` 正在跑
  table-tennis distillation，约 `18.6GB`；GPU `2-7` 均已有较大训练或渲染
  占用；GPU `0` 只有约 `2.1GB` 既有图形/Isaac 进程占用，是唯一适合新
  Pingpong 测试的卡。
- 对本次 PACE-style StrikeQuality reward，建议分两档：
  - 快速测试 / smoke：单卡 GPU `0` 跑 `4096` env。理由是先确认新 reward
    term、checkpoint load、CUDA/EGL/Warp 初始化和指标写入都稳定，避免在已拥挤
    的机器上直接抢占多卡。
  - 稳定后长训：优先恢复到旧 Cross/StrikeQuality 的 `16384` env 双卡配置；
    如果后续空出 GPU `4-7`，再考虑 Impact 线验证过的每卡 `15360` env。
- 启动尝试记录：
  - 直接按 `command.md` 风格启动单卡 `10240` env tmux：
    `pingpong_cross_strike_quality_pace_gpu0_20260706`，run
    `logs/rsl_rl/g1_pingpong_latent_cross_strike_quality/pingpong_cross_strike_quality_pace_from_hit_v3_clean3500_10240env_gpu0_2026-07-06_15-06-37`。
    该 run 只写出 `params/agent.yaml` 和 `params/env.yaml`，tmux 随后退出，尚未
    进入 PPO iteration。
  - sandbox 内前台 `512` env smoke 捕获到：
    `RuntimeError: No CUDA GPUs are available`，Warp 也报告 CUDA driver not
    available。这是当前 sandbox GPU 可见性问题；正式训练需要在有 GPU 权限的
    tmux/host 环境中启动。
  - host 前台 `512` env、`max_iterations=1` smoke 在添加绝对 decoder
    checkpoint override 后通过。该 smoke 确认：
    reward 表包含 `strike_pred_net_clearance=40.0`、
    `strike_pred_landing_inside=80.0`、`strike_post_hit_speed=30.0`，并能从
    Hit V3 clean `model_3500.pt` 加载 actor/critic。
- 当前测试训练：
  - Tmux session: `pingpong_cross_sq_pace_4096_gpu0_20260706`
  - Task: `Mjlab-Pingpong-Cross-StrikeQuality-Unitree-G1`
  - Env count: `4096`
  - GPU: host GPU `0` exposed as `cuda:0`
  - Run:
    `logs/rsl_rl/g1_pingpong_latent_cross_strike_quality/pingpong_cross_strike_quality_pace_from_hit_v3_clean3500_4096env_gpu0_2026-07-06_15-11-36`
  - Launch record:
    `logs/rsl_rl/_codex_launch_records/pingpong_cross_sq_pace_4096_gpu0_20260706/COMMAND.md`
  - Stdout capture:
    `logs/rsl_rl/_codex_launch_records/pingpong_cross_sq_pace_4096_gpu0_20260706/stdout.log`
  - Command used absolute decoder checkpoint:
    `/data0/universe/home_moved/mjlab_latent/logs/rsl_rl/g1_distillation/distill_cloud_unitree_racket_tennis_2026-05-12_09-35-14/model_30000.pt`
    because the default relative decoder path does not exist under the current
    workspace `logs/`.
  - Early live check at 2026-07-06 15:12 confirmed the run was active at
    iteration `3528/43500`, about `43.7k` steps/s. Host GPU `0` total memory
    was `7118MiB / 24564MiB`, with the Pingpong training process using about
    `4950MiB`; remaining GPU `0` memory was mostly existing Isaac/graphics
    processes. Initial PACE-style rewards were nonzero:
    `Episode_Reward/strike_pred_net_clearance ~= 0.40`,
    `strike_pred_landing_inside ~= 0.03`,
    `strike_post_hit_speed ~= 0.24`.
- 正式训练替换：
  - 用户确认约 `20GB` 可用后，停止 `4096` env 测试 session，GPU `0`
    回到约 `2163MiB / 24564MiB` 占用。
  - 正式训练改为单卡 `15360` env（`512*30`），原因是它接近已有
    Cross-Impact 每卡长训验证过的高吞吐档位，同时比单卡 `16384` 多留一点显存
    余量。
  - Tmux session: `pingpong_cross_sq_pace_15360_gpu0_20260706`
  - Run:
    `logs/rsl_rl/g1_pingpong_latent_cross_strike_quality/pingpong_cross_strike_quality_pace_from_hit_v3_clean3500_15360env_gpu0_2026-07-06_15-15-31`
  - Launch record:
    `logs/rsl_rl/_codex_launch_records/pingpong_cross_sq_pace_15360_gpu0_20260706/COMMAND.md`
  - Stdout:
    `logs/rsl_rl/_codex_launch_records/pingpong_cross_sq_pace_15360_gpu0_20260706/stdout.log`
  - 早期检查：iteration `3506/43500`，约 `61k` steps/s；GPU `0` 总占用
    `18918MiB / 24564MiB`，训练进程约 `16750MiB`，剩余约 `5.6GB`。
  - 已清理本轮无关记录：失败的 `10240` env params-only run、失败的
    `4096` env run、停止的 `4096` env test run、两个 `512` env smoke run、
    4096 test launch record，以及对应的两个 smoke/test W&B offline run
    `offline-run-20260706_151055-b72qo23t` 和
    `offline-run-20260706_151140-68zveryz`。正式 `15360` run 和
    `offline-run-20260706_151540-f5tc872d` 保留。

## 2026-07-05 - Table-Tennis Latent Analysis Diagnostic

### 目标

为 table-tennis distillation checkpoint 新增离线 latent-space 诊断，固定分析
`model_30000.pt`，覆盖 `artifacts/table_tennis` 下 train、held-out
`test_001` 和 diagnostic `zhengshou_002_badend` motion，不修改 Pingpong 默认
decoder。

### 计划

- [x] 新建 `latent_analysis/`，包含主脚本、README 和输出目录。
- [x] 解析 distillation TensorBoard event，生成 final/best/last100 指标摘要。
- [x] 通过 distillation env 进行 CPU teacher-forced reference-state 采样，提取
  posterior/prior latent、teacher action、prior/posterior reconstruction。
- [x] 输出 latent 统计、重构误差、PCA/诊断图和 verdict report。
- [x] 运行脚本生成本地结果并做窄验证。

### 实现记录

- `latent_analysis/analyze_latent_space.py` 默认使用
  `/data0/universe/home_moved/mjlab_latent/logs/rsl_rl/g1_distillation_table_tennis/table_tennis_distill_v1_46080env_from_tracking18000_2026-07-03_10-14-26/model_30000.pt`。
- 采样方式是 teacher-forced reference-state probe：平衡设置各 motion/frame，
  warmup 后用 tracking teacher action 推进一步，再对同一 actor obs 计算
  `q(z|state,target)`、`p(z|state)`、posterior/prior decode action。
- 输出写入 `latent_analysis/outputs/`，包括 JSON、CSV、PNG、`report.md` 和
  `report.html`。

### 诊断结果

- 正式命令：
  `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run python latent_analysis/analyze_latent_space.py --checkpoint /data0/universe/home_moved/mjlab_latent/logs/rsl_rl/g1_distillation_table_tennis/table_tennis_distill_v1_46080env_from_tracking18000_2026-07-03_10-14-26/model_30000.pt --motion-root artifacts/table_tennis --output-dir latent_analysis/outputs --samples-per-motion 1024`
- 采样结果：13 个 motion 各 1024 个样本，总计 13312。
- Verdict: `not_ready`。核心原因是 latent 使用塌缩：active dims `0/16`，
  posterior/prior mean L2 `0.00735`，KL mean `0.000346`，posterior/prior
  action MSE 几乎相同。
- Teacher-action reconstruction：global prior MSE `0.0691`，train prior MSE
  `0.0714`，held-out `test_001` prior MSE `0.0527`，diagnostic
  `zhengshou_002_badend` prior MSE `0.0607`。
- TensorBoard final metrics still look much lower：final loss `0.01532`，
  final prior_action_loss `0.01531`，last100 prior_action_loss `0.01578`。
  这说明训练日志上的 on-policy/rollout 分布与 reference-state 离线 probe
  存在差异，需要继续做闭环 play 或修改蒸馏目标后再选 decoder。

## 2026-07-05 - Tennis Default Latent Checkpoint Diagnostic

### 目标

按 table-tennis latent diagnostic 同样流程，分析当前 Tennis/Pingpong 默认引用的
tennis latent decoder checkpoint，不修改默认 decoder。

### 实现记录

- 将 `latent_analysis/analyze_latent_space.py` 泛化为可指定
  `--task-id`、`--analysis-label` 和 `--motion-pattern`，默认 table-tennis 行为
  保持不变。
- Tennis 分析对象：
  `/data0/universe/home_moved/mjlab_latent/logs/rsl_rl/g1_distillation/distill_cloud_unitree_racket_tennis_2026-05-12_09-35-14/model_30000.pt`。
- 对应 task：`Mjlab-Distill-Flat-Unitree-G1`；motion：
  `artifacts/tennis_random_*/motion.npz`。

### 诊断结果

- 正式命令：
  `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run python latent_analysis/analyze_latent_space.py --analysis-label tennis-default --task-id Mjlab-Distill-Flat-Unitree-G1 --checkpoint /data0/universe/home_moved/mjlab_latent/logs/rsl_rl/g1_distillation/distill_cloud_unitree_racket_tennis_2026-05-12_09-35-14/model_30000.pt --motion-root artifacts --motion-pattern 'tennis_random_*/motion.npz' --output-dir latent_analysis/outputs_tennis --samples-per-motion 1024`
- 输出：`latent_analysis/outputs_tennis/`。
- 采样结果：4 个 tennis motion 各 1024 个样本，总计 4096。
- Verdict: `not_ready`。latent 比 table-tennis 更活跃，active dims `8/16`，
  但 prior/posterior 对齐差，KL mean `6.18`，posterior/prior mean L2 `1.16`。
- Teacher-action reconstruction：global posterior MSE `0.0835`，global prior
  MSE `0.1033`。各 motion prior MSE：`tennis_random_001=0.1078`，
  `002=0.1051`，`003=0.1131`，`004=0.0870`。
- TensorBoard final metrics：loss `0.01669`，action_loss `0.01483`，
  prior_action_loss `0.01719`，kl_loss `0.3721`。离线 reference-state probe
  明显更严格，说明该 decoder 虽然部署上可被高层使用，但不应被当作
  “latent prior 压缩良好”的 checkpoint。

## 2026-07-05 - Table-Tennis Distillation V2 KL-Aligned Launch

### 目标

重新启动 table-tennis low-level distillation，只对齐 tennis 默认 decoder 的 KL
schedule，不修改 teacher、motion、asset、state/target terms 或 Pingpong 默认
decoder。

### 启动记录

- Tmux session:
  `table_tennis_distill_v2_kl_aligned_gpu1_20260705`
- Task: `Mjlab-Distill-TableTennis-Unitree-G1`
- Host GPU: `1` exposed as `cuda:0`
- Env count: `46080`
- Teacher checkpoint:
  `/data0/universe/home_moved/mjlab_latent/logs/rsl_rl/g1_tracking_table_tennis/table_tennis_tracking_v1_18432env_gpu2_2026-07-02_17-39-35/model_18000.pt`
- Run:
  `logs/rsl_rl/g1_distillation_table_tennis/table_tennis_distill_v2_46080env_kl_aligned_from_tracking18000_gpu1_2026-07-05_23-43-32`
- Command:
  `CUDA_VISIBLE_DEVICES=1 UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run train Mjlab-Distill-TableTennis-Unitree-G1 --env.scene.num-envs 46080 --gpu-ids '[0]' --agent.teacher-checkpoint /data0/universe/home_moved/mjlab_latent/logs/rsl_rl/g1_tracking_table_tennis/table_tennis_tracking_v1_18432env_gpu2_2026-07-02_17-39-35/model_18000.pt --agent.kl-loss-weight 0.001 --agent.kl-loss-weight-end 0.005 --agent.kl-loss-anneal-start 2500 --agent.kl-loss-anneal-end 10000 --agent.experiment-name g1_distillation_table_tennis --agent.run-name table_tennis_distill_v2_46080env_kl_aligned_from_tracking18000_gpu1`

### 参数对齐

- 对齐 tennis 的 KL schedule：
  `kl_loss_weight=0.001`,
  `kl_loss_weight_end=0.005`,
  `kl_loss_anneal_start=2500`,
  `kl_loss_anneal_end=10000`。
- 保持 table-tennis teacher、motion、robot asset、state terms、target terms、
  latent dim、batch size、buffer 和 teacher-action schedule 不变。
- Resolved `params/agent.yaml` confirmed the above overrides.

### 初始状态

- Sandbox 内直接跑 CUDA smoke failed: `RuntimeError: No CUDA GPUs are available`。
  原因是 sandbox 内 `torch` 看不到 CUDA；正式训练已通过 sandbox 外 tmux 启动。
  失败 smoke run 目录：
  `logs/rsl_rl/g1_distillation_table_tennis_smoke/smoke_kl_aligned_cli_2026-07-05_23-43-00`
  不是有效训练。
- `iter 0`: loss `2.05281`, action `2.04871`, prior_action `2.07290`,
  KL `4.09406`, `kl_w=0.001`, buffer `737280`。
- `iter 20`: loss `0.01336`, action `0.01320`, prior_action `0.01302`,
  KL `0.16503`, `kl_w=0.001`, buffer `1048576`。
- 训练初期显存约 `18.6GB` on host GPU 1。

## 2026-07-03 - Table-Tennis Distillation Launch

### 目标

用 table-tennis tracking teacher `model_18000.pt` 启动 low-level latent
distillation，生成后续可供 Pingpong frozen decoder action 使用的 decoder
checkpoint。用户要求将环境数放大到初始 distillation 的 `2.5x`，即
`18432 -> 46080`。

### 启动记录

- Active tmux session: `table_tennis_distill_v1_46080env_gpu0_20260703`
- Task: `Mjlab-Distill-TableTennis-Unitree-G1`
- GPU: host GPU 0 exposed as `cuda:0`
- Env count: `46080`
- Teacher checkpoint:
  `logs/rsl_rl/g1_tracking_table_tennis/table_tennis_tracking_v1_18432env_gpu2_2026-07-02_17-39-35/model_18000.pt`
- Run name: `table_tennis_distill_v1_46080env_from_tracking18000`
- Output:
  `logs/rsl_rl/g1_distillation_table_tennis/table_tennis_distill_v1_46080env_from_tracking18000_2026-07-03_10-14-26`
- Command:
  `CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run train Mjlab-Distill-TableTennis-Unitree-G1 --env.scene.num-envs 46080 --gpu-ids "[0]" --agent.teacher-checkpoint logs/rsl_rl/g1_tracking_table_tennis/table_tennis_tracking_v1_18432env_gpu2_2026-07-02_17-39-35/model_18000.pt --agent.teacher-action-prob 1.0 --agent.experiment-name g1_distillation_table_tennis --agent.run-name table_tennis_distill_v1_46080env_from_tracking18000`

### 初始状态

- Resolved config confirms `teacher_task_id=Mjlab-Tracking-TableTennis-Unitree-G1`,
  `teacher_action_prob=1.0`, action shape `29`, and state terms
  `base_lin_vel, base_ang_vel, joint_pos, joint_vel, actions`.
- `model_0.pt` and an ONNX export were written.
- At iteration `10/29999`, loss dropped from `2.069` to `0.020`, replay buffer
  filled to `1048576`, and GPU 0 was using about `18.6GB`.
- The previous `18432`-env trial
  `table_tennis_distill_v1_from_tracking18000_2026-07-03_10-10-17` was stopped
  at about iteration `140` before `model_250.pt`; it has only `model_0.pt` and
  logs.
- Do not update Pingpong `DEFAULT_DECODER_CHECKPOINT` to the tracking teacher.
  Switch Pingpong action only after a distillation decoder checkpoint is selected
  and validated.

## 2026-07-02 - Table-Tennis Tracking 18432-Env Relaunch

### 目标

按用户要求将 table-tennis tracking 训练环境数改为 `18*1024=18432`，重新训练。

### 启动记录

- 已停止旧 session `table_tennis_tracking_v1_gpu2_20260702`。
- 新 tmux session: `table_tennis_tracking_v1_18432env_gpu2_20260702`
- Task: `Mjlab-Tracking-TableTennis-Unitree-G1`
- GPU: host GPU 2 exposed as `cuda:0`
- Env count: `18432`
- Run name: `table_tennis_tracking_v1_18432env_gpu2`
- Output:
  `logs/rsl_rl/g1_tracking_table_tennis/table_tennis_tracking_v1_18432env_gpu2_2026-07-02_17-39-35`
- Command:
  `CUDA_VISIBLE_DEVICES=2 UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run train Mjlab-Tracking-TableTennis-Unitree-G1 --env.scene.num-envs 18432 --gpu-ids "[0]" --agent.experiment-name g1_tracking_table_tennis --agent.run-name table_tennis_tracking_v1_18432env_gpu2`

### 初始状态

- Environment table confirmed `Number of environments | 18432`.
- Training reached iteration `1/30000`; early speed after warmup was about
  `190k steps/s`.
- Initial terminations are still dominated by `ee_body_pos`, so this run should
  be watched before selecting a teacher checkpoint.

### Held-out test_001 check

- Checkpoint:
  `logs/rsl_rl/g1_tracking_table_tennis/table_tennis_tracking_v1_18432env_gpu2_2026-07-02_17-39-35/model_1500.pt`
- Motion:
  `artifacts/table_tennis/test_001/motion.npz`
- Rollout: `600` control steps, no termination or truncation.
- Mean errors: `anchor_pos=0.033m`, `anchor_rot=0.041rad`,
  `body_pos=0.026m`, `body_rot=0.145rad`, `joint_pos_l2=0.787rad`,
  `body_lin_vel=0.091`, `body_ang_vel=0.372`.
- Preview:
  `outputs/table_tennis_test001_tracking_model1500.mp4`
- 2026-07-03 follow-up with `model_17500.pt`: `600` control steps, no
  termination or truncation. Mean errors improved to `body_pos=0.019m`,
  `body_rot=0.091rad`, `joint_pos_l2=0.414rad`, with `anchor_pos=0.034m` and
  `anchor_rot=0.044rad`.
- 2026-07-03 video with `model_18000.pt` on `test_001`:
  `outputs/table_tennis_test001_tracking_model18000_h264.mp4`. It is
  H.264/yuv420p, `960x540`, `50fps`, `12s`.

## 2026-07-02 - Table-Tennis Tracking Training Launch

### 目标

启动 table-tennis low-level tracking teacher 正式训练。

### 启动记录

- Tmux session: `table_tennis_tracking_v1_gpu2_20260702`
- Task: `Mjlab-Tracking-TableTennis-Unitree-G1`
- GPU: host GPU 2 exposed as `cuda:0`
- Env count: `4096`
- Run name: `table_tennis_tracking_v1_gpu2`
- Output:
  `logs/rsl_rl/g1_tracking_table_tennis/table_tennis_tracking_v1_gpu2_2026-07-02_17-34-02`
- Command:
  `CUDA_VISIBLE_DEVICES=2 UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run train Mjlab-Tracking-TableTennis-Unitree-G1 --env.scene.num-envs 4096 --gpu-ids "[0]" --agent.experiment-name g1_tracking_table_tennis --agent.run-name table_tennis_tracking_v1_gpu2`

### 初始状态

- Preflight confirmed 11 local training motions under `artifacts/table_tennis`,
  with no missing paths.
- Training reached iteration `48/30000` within the first minute. Initial
  tracking errors are still high, with very short episode length and many
  `ee_body_pos` terminations, so this should be watched before assuming the
  teacher is healthy.

## 2026-07-02 - Table-Tennis Low-Level Chain

### 目标

新增乒乓球 low-level 数据链路：table pkl 转 tracking motion、table-tennis
tracking teacher task、table-tennis distillation task，同时保持 tennis 默认任务不变。

### 实现记录

- 新增 `uv run -m mjlab.scripts.table_pkl_to_npz`，读取 `table_data/*.bvh_wxy.pkl`
  并输出 `artifacts/table_tennis/<motion>/motion.npz`。
- 转换器按 `root_pos`、xyzw `root_rot`、29 维 `dof_pos`、`fps` 加载数据，
  重采样到目标 fps，经当前 G1 pingpong-paddle MuJoCo 模型 replay 后保存
  tracking 兼容字段。
- 新增共享 `get_g1_w_pingpong_paddle_robot_cfg()` helper，供 Pingpong、
  Tracking table-tennis 和 Distillation table-tennis 复用；当前保持缩放原
  `tennis_racket` mesh 的模型路线，不使用单独 primitive 乒乓拍。
- 注册 `Mjlab-Tracking-TableTennis-Unitree-G1` 和
  `Mjlab-Distill-TableTennis-Unitree-G1`。tracking 训练 split 排除 `test_001`
  和 `zhengshou_002_badend`；distill runner 指向 table-tennis tracking teacher
  且默认要求通过 CLI 提供 teacher checkpoint。

### 验证记录

- `FORCE_CPU=1 UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run pytest tests/test_tracking_task.py tests/test_pingpong_task.py tests/test_table_pkl_to_npz.py -q`
  通过。
- `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run ty check src/mjlab/scripts src/mjlab/tasks/tracking src/mjlab/tasks/distillation src/mjlab/asset_zoo/robots/unitree_g1_w_pingpong_paddle.py tests/test_tracking_task.py tests/test_table_pkl_to_npz.py`
  通过。

## 2026-07-02 - Revert Pingpong Primitive Paddle Geometry

### 目标

撤回把 G1 pingpong 持拍模型改成单独 primitive 乒乓拍的尝试，恢复到参考原
tennis 持拍模型的缩放 mesh 路线。

### 实现记录

- `get_g1_w_pingpong_paddle_spec()` 恢复为缩放 `tennis_racket` visual mesh，
  并继续使用 `pingpong_paddle_collision` 作为拍面碰撞 proxy。
- 移除 primitive 版本中的 ellipsoid 拍面、独立可见 handle 和隐藏 tennis
  mesh 语义；`pingpong_paddle_handle_collision` 仍是不计分的透明手柄碰撞体。
- `tests/test_pingpong_task.py` 恢复为检查 scaled mesh 契约，并显式确认不存在
  `pingpong_paddle_handle_visual`。
- `docs/source/changelog.rst` 和 `summary.html` 移除了 primitive paddle 记录。

### 验证记录

- `FORCE_CPU=1 UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run pytest tests/test_pingpong_task.py tests/test_tracking_task.py -q`
  通过：21 passed。
- `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run ty check src/mjlab/asset_zoo/robots/unitree_g1_w_pingpong_paddle.py src/mjlab/tasks/pingpong/config/g1/env_cfgs.py tests/test_pingpong_task.py`
  通过。
- 对比脚本确认 pingpong helper 重新显示 `pingpong_paddle_visual` mesh，并只保留
  `pingpong_paddle_collision` 与 `pingpong_paddle_handle_collision` 两个 paddle
  collision proxy。

## 2026-07-02 - Pingpong Cross Reward Simplification

### 目标

把 Pingpong Cross reward 从 strike/pred/impact 多项 shaping 收敛为用户确认的
最小结构：预击球、后击球、真实 sparse success 和基础保护。

### 计划

- [x] 用预测击球点替换 Cross 中追当前球的预击球 reward。
- [x] 让后击球 `x_progress` 和 `velocity_direction` 使用相同权重和横向 gate。
- [x] 从 Cross-StrikeQuality/Impact reward 中移除 pred landing/net 和 impact
  paddle shaping，保留这些信息为 metrics。
- [x] 放松动作变化正则，避免过早压制击球加速。

### 实现记录

- Cross 系列移除 `approach_ball` 和 `paddle_towards_ball`，新增
  `hit_point=10`，通过 `paddle_to_predicted_hit_point_dense` 奖励球拍中心靠近
  `ball_predicted_edge_hit_point_b` 给出的 post-bounce 击球点。
- `post_hit_x_progress` 和 `post_hit_ball_velocity_direction` 都设为 `120`，都使用
  `lateral_speed_std=0.8` 抑制横飞出界的回球。`x_progress` 仍保留较强权重，但
  不再单独奖励大横向速度。
- `strike_outgoing_ball_velocity`、`strike_pred_net_clearance`、
  `strike_pred_landing_inside`、`strike_post_hit_speed`、
  `impact_paddle_to_target_velocity`、`impact_paddle_normal_alignment`、
  `impact_paddle_normal_velocity`、`impact_contact_centering` 和
  `followthrough_velocity` 不再作为 Cross/Impact reward。Impact task 继续保留
  impact-window metrics 作为诊断。
- `paddle_hit_event` 调整为 `40`，`opponent_table_bounce_event` 调整为 `1200`。
  Cross loose regularization 调整为 `latent_action_rate_l2=-0.0025`、
  `low_level_action_rate_l2=0.0`，其余保护项保持不变。

### 验证记录

- `UV_CACHE_DIR=/tmp/uv-cache FORCE_CPU=1 uv run pytest tests/test_pingpong_task.py tests/test_pingpong_state.py -q`
  通过：20 passed。
- `UV_CACHE_DIR=/tmp/uv-cache uv run ty check src/mjlab/tasks/pingpong tests/test_pingpong_task.py tests/test_pingpong_state.py`
  通过。

## 2026-07-02 - Pingpong Paddle-Ball Contact 标定

### 目标

用本地 `contact_test` 最小场景标定 pingpong paddle-ball 碰撞，再把实验支持的
显式 contact pair 写回 pingpong scene，避免继续依赖当前偏软的 geom 混合接触。

### 计划

- [x] 在 `contact_test/` 中新增 paddle-ball sweep 脚本和 JSON 配置。
- [x] 跑默认计划 sweep，确认原候选范围是否足够。
- [x] 跑 extended/focused sweep，寻找能达到 `e_n ~= 0.55` 的参数。
- [x] 只在 pingpong G1 scene 注入 explicit pair，不修改共享 G1-with-racket 资产。
- [x] 增加 compiled-model pair 参数测试和 sensor 语义回归。

### 实现记录

- 新增本地 ignored 实验入口：
  `contact_test/run_pingpong_paddle_contact_sweep.py`，配置在
  `contact_test/configs/paddle_ball_sweep*.json`。脚本读取当前 pingpong 球半径/质量、
  拍面尺寸和 env timestep，构建固定薄圆柱 paddle + 自由 pingpong ball 的最小 MuJoCo
  scene，并输出 `summary.csv` / `summary.json`。
- 默认计划 sweep 输出：
  `contact_test/results/2026-07-02_14-27-55`。在 `damping>=0.55` 范围内仍偏软，
  最好 `2/3/5m/s` 中位 `e_n` 约 `0.41`，未达到 `0.45~0.70`。
- Extended/focused sweep 输出：
  `contact_test/results/2026-07-02_14-28-35` 和
  `contact_test/results/2026-07-02_14-31-04`。当前 baseline 中位 `e_n` 约 `0.158`；
  写回候选为 `solref=(0.011, 0.40)`、`friction=(0.08, 0.002, 0.0001)`、
  `solimp=(0.93, 0.98, 0.001)`，margin 后续按提前接触风险折中为 `0.010`。
- Focused best row：`2/3/5/8m/s` 的 `e_n` 中位约
  `0.555/0.609/0.490/0.474`，`5m/s` 最大穿透约 `4.687mm`，`8m/s` 全相位分离。
  注意：满足该严格稳定行需要较大 margin。主配置改用 `margin=0.010` 以减轻提前
  接触；该折中预计会保留中等回弹，但不再满足 `5m/s` 穿透和 `8m/s` 全相位分离的
  严格标定标准。若视频仍显示提前接触过强，下一步应在继续降低 margin 与更小
  physics timestep 之间权衡。
- 主配置新增 `add_pingpong_paddle_ball_contact_pair()`，只通过 G1 pingpong
  `_apply_g1_pingpong_common()` 的 `cfg.scene.spec_fn` 注入；tennis/tracking/distillation
  共用 racket asset 不变。

### 验证记录

- `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run python contact_test/run_pingpong_paddle_contact_sweep.py`
- `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run python contact_test/run_pingpong_paddle_contact_sweep.py --config contact_test/configs/paddle_ball_sweep_extended.json --max-print-rows 16`
- `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run python contact_test/run_pingpong_paddle_contact_sweep.py --config contact_test/configs/paddle_ball_margin_focus.json --max-print-rows 12`
- `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run pytest tests/test_pingpong_task.py::test_pingpong_paddle_ball_explicit_contact_pair tests/test_pingpong_task.py::test_pingpong_paddle_handle_collision_does_not_score -q`
  通过：2 passed。
- `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run pytest tests/test_pingpong_task.py tests/test_pingpong_state.py -q`
  通过：20 passed。
- `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run pytest tests/test_ball_sport_geometry.py -q`
  通过：1 passed。
- `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run ruff check src/mjlab/tasks/pingpong/pingpong_env_cfg.py src/mjlab/tasks/pingpong/config/g1/env_cfgs.py tests/test_pingpong_task.py`
  通过。
- `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run ty check src/mjlab/tasks/pingpong/pingpong_env_cfg.py src/mjlab/tasks/pingpong/config/g1/env_cfgs.py tests/test_pingpong_task.py`
  通过。

## 2026-07-02 - Pingpong Cross Impact-Window Guidance

### 目标

为 Pingpong Cross 新增一条击球窗口内的球拍行为引导实验线，避免策略只依赖
post-hit 球行为奖励学习“挡球”，而是在 legal hit 前后直接引导球拍中心、法向、
速度和 follow-through。

### 计划

- [x] 在 `PingpongRallyState` 中缓存 impact-window 几何和速度诊断。
- [x] 新增 impact-window paddle behavior rewards。
- [x] 新增 `Mjlab-Pingpong-Cross-Impact-Unitree-G1` ablation task。
- [x] 注册 G1 env / PPO config，并保持旧 Cross / StrikeQuality reward keys 不变。
- [x] 增加状态机和配置测试，运行窄验证。

### 预期改动

新增 Impact task 复用 Cross loose regularization 和 StrikeQuality post-hit rewards，
额外启用 `impact_paddle_to_target_velocity`、
`impact_paddle_normal_alignment`、`impact_paddle_normal_velocity`、
`impact_contact_centering`、`followthrough_velocity`。默认 checkpoint 仍不写死，
训练时通过 `--agent.load-checkpoint-file` 显式传入。

### 进展记录

- `PingpongRallyState` 增加 impact-window active gate、固定保守出球目标、球拍
  center/normal/velocity、ball state、desired outgoing direction、center
  distance、velocity-to-target、velocity-along-normal、normal-to-target 和
  follow-through velocity 诊断。
- Impact active window 定义为自方落台后、首次 legal hit/fault 前、球拍近场；
  legal hit step 总是纳入窗口，避免几何中心距离门控漏掉真实接触。
- 新任务 ID：`Mjlab-Pingpong-Cross-Impact-Unitree-G1`。
- 新 experiment/run：`g1_pingpong_latent_cross_impact` /
  `pingpong_cross_impact_from_hit`。
- 新 metrics：`impact/window_active`、`impact/window_count`、
  `impact/velocity_to_target`、`impact/velocity_along_normal`、
  `impact/normal_to_target`、`impact/center_distance`、
  `impact/followthrough_velocity`。
- 2026-07-02 启动训练时先用 GPU `4,5,6,7`、每卡 `10240` env 做 smoke；
  smoke 暴露旧 reward 签名未接收新增 state 参数的问题。已让
  `approach_ball` 和 `paddle_towards_ball` 接收共享 state `**params`，并新增
  签名兼容测试。
- 用户要求停止并清理每卡 `10240` env 的长训；已删除 run 目录
  `logs/rsl_rl/g1_pingpong_latent_cross_impact/pingpong_cross_impact_hitbest_10240pergpu_gpu4_7_20260702_105707_2026-07-02_10-57-35`
  和 launch record
  `logs/rsl_rl/_codex_launch_records/pingpong_cross_impact_4567_20260702_105707`。
- 当前活跃长训改为 GPU `4,5,6,7`、每卡 `15360` env（`512*30`，总
  `61440` env），tmux
  `pingpong_cross_impact_4567_15360_20260702_110128`，run:
  `logs/rsl_rl/g1_pingpong_latent_cross_impact/pingpong_cross_impact_hitbest_15360pergpu_gpu4_7_20260702_110128_2026-07-02_11-01-52`。
  启动后首批指标正常写入，`impact/window_active` 约 `0.12`，
  `impact/velocity_to_target` 约 `0.36`。

### 验证记录

- `UV_CACHE_DIR=/tmp/uv-cache FORCE_CPU=1 uv run pytest tests/test_pingpong_state.py tests/test_pingpong_task.py -q`
  通过：19 passed。
- `UV_CACHE_DIR=/tmp/uv-cache uv run ty check src/mjlab/tasks/pingpong tests/test_pingpong_state.py tests/test_pingpong_task.py`
  通过。
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/mjlab/tasks/pingpong tests/test_pingpong_state.py tests/test_pingpong_task.py`
  通过。

### 下一步

先用 4096 env 做短训 smoke，确认 `impact/window_active`、`impact/velocity_to_target`、
`impact/velocity_along_normal` 和 `impact/normal_to_target` 有非零学习信号，再上
16384 env 长训。最终仍以 `crossed_net_count`、`opponent_table_bounce_count` 和
`legal_return_count` 是否从 0 出现为判断标准。

## 2026-06-20 18:13 - Pingpong V1 Task

### 目标

在当前 G1 网球 latent-control 基线上新增乒乓球初版任务，目标是实现合法单拍回球：
来球在己方台面弹起，机器人击球，球过网，并第一次落在对方台面界内。

### 计划

- [x] 从当前工作区创建并切换到 `pingpang` 分支。
- [x] 新增程序化乒乓球场景资产：球桌、球网、乒乓球。
- [x] 新增 G1 乒乓球拍 spec wrapper，复用持拍 G1 资产但不修改 shared tennis XML。
- [x] 新增 feeder、合法回球状态机、奖励、终止和 metrics。
- [x] 注册 `Mjlab-Pingpong-Hit-Unitree-G1` 与 `Mjlab-Pingpong-Return-Unitree-G1`。
- [x] 增加窄测试并运行验证。

### 预期改动

新增 `src/mjlab/tasks/pingpong/` 任务族和 `tests/test_pingpong_*.py`。V1 复用 frozen
tennis latent decoder，不新增 motion data 或重新训练 distillation decoder。

### 进展记录

- 分支 `pingpang` 已创建并切换成功。
- `Hit` 任务使用同一个合法回球状态机，但成功项是第一次合法 post-bounce paddle hit。
- `Return` 任务在 `Hit` 基础上增加 post-hit x progress、velocity direction、crossed-net 和 opponent-table-bounce 奖励，并以 `legal_return_success` 终止。
- `list-envs` 输出中已出现两个 Pingpong 任务 ID；该命令返回码为 26，但表格输出完整。

### 问题记录

#### 问题 1：沙盒无法写 uv cache

现象：`uv run pytest` 和 `uv run ty check` 初次执行失败，提示 `/home/universe/.cache/uv` 为只读。
可能原因：当前沙盒只允许工作区和 `/tmp` 写入，uv 默认 cache 在用户 home 下。
验证方式：按授权流程使用 escalated 权限重跑验证命令。
验证结果：`tests/test_pingpong_state.py tests/test_pingpong_task.py` 通过，类型检查和 Ruff 检查通过。
状态：Resolved
最终总结：以后本仓库中 `uv run ...` 若遇到 cache 只读，可直接按授权流程重跑。

## 2026-06-22 16:10 - Pingpong Version 1：追到球

### 目标

记录当前乒乓球初版训练成果，并把代码状态作为 `version 1: 追到球` 提交到 git。

### 训练记录

- Run: `logs/rsl_rl/g1_pingpong_latent_hit/pingpong_hit_8192env_gpu2_4_2026-06-21_23-07-11`
- Task: `Mjlab-Pingpong-Hit-Unitree-G1`
- Env count: `8192`
- GPUs: `2,4`
- Latest observed checkpoint: `model_19000.pt`
- Latest observed iteration: `19210/33000`

### 当前结论

- 机器人已经达到 V1 里程碑：能追到球并完成第一拍击球。
- Curriculum 已到 stage `5`，`self_table_bounce_count` 基本为 `1.0`。
- 最新日志附近 `paddle_hit_count` 约 `0.61-0.62`，curriculum `success_rate` 约 `0.60-0.62`。
- `crossed_net_count`、`opponent_table_bounce_count` 和 `legal_return_count` 仍为 `0`，这符合当前 `Hit` 任务语义；合法回球应在 `Return` 任务继续训练。

### 风险和下一步

- `Mean action std` 已升到约 `694`，说明探索噪声过大；继续原配置训练可能不会显著提升稳定性。
- 下一步建议用 `model_19000.pt` 做 play/video 验证，并在新实验中降低 entropy 或增加 action std 约束。

## 2026-06-22 17:00 - Pingpong V2：小球拍、避桌、端线二跳

### 目标

在 V1 已经能追到球的基础上，提高任务真实性和后续训练质量：球拍视觉与碰撞一起缩小，
机器人碰桌只惩罚不终止，并让来球优先形成机器人侧端线附近的二跳目标。

### 实现记录

- G1 pingpong spec wrapper 缩放 `tennis_racket` visual mesh，重命名为 `pingpong_paddle_visual`。
- `pingpong_paddle_collision` 半径改为 `0.065`，`pingpong_paddle_center` 按同一缩放比例对齐。
- 新增 `robot_table_contact` contact sensor、同名 reward penalty、`robot_table_contact_count` metric。
- `TableTennisFeederCfg.target_mode` 默认改为 `edge_second_bounce`，通过反解第一跳让近似第二跳落在机器人侧端线附近。
- 新增 `ball_predicted_edge_hit_point_b`，替代 actor/critic 的 `predicted_hit_point`，保持 `(x, y, z, t)` 维度不变。

### 验证

- `uv run pytest tests/test_pingpong_state.py tests/test_pingpong_task.py tests/test_pingpong_observations.py -q`
- `uv run ruff check src/mjlab/tasks/pingpong tests/test_pingpong_state.py tests/test_pingpong_task.py tests/test_pingpong_observations.py`
- `uv run ty check src/mjlab/tasks/pingpong tests/test_pingpong_task.py tests/test_pingpong_state.py tests/test_pingpong_observations.py`

### 后续建议

- 重新训练时建议从 V1 checkpoint warm-start，但降低 entropy 或约束 action std。
- 训练日志需要重点看 `robot_table_contact_count`、`paddle_hit_count`、`success_rate` 和 `Mean action std`。

## 2026-06-23 00:30 - Ball-Sport Geometry Provider 解耦

### 目标

整理乒乓发球 provider：把“生成 first-bounce 轨迹”的通用逻辑和“当前阶段只发长球”
的 profile 约束分离，并在 pingpong/tennis 上方抽象出通用球类运动场地几何。

### 实现记录

- `TableTennisFeederCfg` 移除 `edge_second_bounce` 语义，改为 scene-driven first-bounce
  sampler：采样敌方半场桌面发球点、我方半场首次弹跳点、逐样本过网 `vz_min`，再反解
  `vx/vy`。
- 新增 `BallSportGeometry` / `resolve_ball_sport_geometry`，从 active scene 解析
  opponent/self half、net、ball radius、可选 bounce/landing plane 以及 surface/ball/net
  contact metadata。Pingpong provider 面向该抽象工作，不再直接依赖 table top 字段。
- 新增 `TrajectoryCheckCfg` 和 `check_candidate_trajectory`，通过配置表达过网、首跳、
  底线穿越、二跳出 self half 等约束；当前 hit profile 开启长球约束。
- `ball_predicted_edge_hit_point_b` 改为使用同一套 scene geometry，预测弹后轨迹与
  机器人侧底线平面的交点，不再默认使用桌内二跳点。
- 新增 tennis geometry smoke test，确认同一 resolver 能用 `court_visual`、
  `tennis_net_collision` 和 `tennis_ball` 解析网球场，不改变 tennis provider 行为。
- 根据真机部署语义，Pingpong actor 的 `ball_pos_window` 从球相对球拍改为
  `ball_position_b`：球心在 robot base frame 下的 10 帧位置历史；critic 的
  `paddle_to_ball` privileged 辅助项暂时保留。

### 验证

- `uv run pytest tests/test_ball_sport_geometry.py tests/test_pingpong_provider.py tests/test_pingpong_observations.py tests/test_pingpong_task.py tests/test_pingpong_state.py -q`
- `uv run ruff check src/mjlab/tasks/ball_sports src/mjlab/tasks/pingpong tests/test_ball_sport_geometry.py tests/test_pingpong_provider.py tests/test_pingpong_task.py tests/test_pingpong_state.py tests/test_pingpong_observations.py`
- `uv run ty check src/mjlab/tasks/ball_sports src/mjlab/tasks/pingpong tests/test_ball_sport_geometry.py tests/test_pingpong_provider.py tests/test_pingpong_task.py tests/test_pingpong_state.py tests/test_pingpong_observations.py`

### 问题记录

#### 问题 1：二跳目标语义错误

现象：上一版 provider 通过 `edge_second_bounce` 反解首跳点，让预测第二次桌面弹跳落在
端线附近。
可能原因：把“弹后轨迹与桌面边沿平面的交点”误解成“第二次桌面弹跳点”。
验证方式：阅读 provider 公式和 state fault 逻辑。
验证结果：原实现不会鼓励多跳继续打，但发球生成确实朝桌内二跳目标采样。
状态：Resolved
最终总结：provider 现在只负责 first-bounce 轨迹；长球由 checker 约束为弹后过底线且二跳出 self half。

## 2026-06-23 16:51 - Pingpong Contact Test 碰撞标定

### 目标

创建本地 `contact_test/` 实验目录，按真实乒乓球无旋转反弹标定 ball/table contact，
保留实验记录和视频，并将最佳参数回写到主 pingpong 配置。

### 实现记录

- `.gitignore` 新增 `/contact_test/`，实验脚本、CSV、JSON 和 mp4 均保留本地但不进入 git。
- `contact_test/run_pingpong_contact_sweep.py` 使用轻量 ball/table scene 做无旋转 drop/oblique
  sweep；contact test 中关闭 net collision，只标定桌面碰撞，视频使用
  `MUJOCO_GL=egl uv run python ... --video True` 方式生成。
- 最佳组为 `stable_088_low_friction`，输出目录
  `contact_test/results/2026-06-23_16-51-41`，包含 18 个 mp4、summary CSV/JSON、
  per-trial metrics 和轨迹 CSV。
- 最佳指标：mean `e_z=0.9007`，mean horizontal retention `0.9374`。
- 主配置回写：ball/table friction `(0.04, 0.002, 0.0001)`，
  solref `(0.002, 0.50)`，solimp `(0.93, 0.98, 0.001)`；
  provider/observation post-bounce scales 使用 `horizontal=0.94`、`vertical=0.90`。

### 验证

- `MUJOCO_GL=egl uv run python contact_test/run_pingpong_contact_sweep.py --config contact_test/configs/contact_sweep.yaml --video True`
- `uv run pytest tests/test_pingpong_provider.py tests/test_pingpong_observations.py tests/test_pingpong_task.py tests/test_pingpong_state.py -q`
- `uv run ruff check src/mjlab/tasks/pingpong tests/test_pingpong_provider.py tests/test_pingpong_observations.py tests/test_pingpong_task.py tests/test_pingpong_state.py`
- `uv run ty check src/mjlab/tasks/pingpong tests/test_pingpong_provider.py tests/test_pingpong_observations.py tests/test_pingpong_task.py tests/test_pingpong_state.py`

## 2026-06-24 - Pingpong Reward Hacking 修正

### 目标

修正 Hit 训练中机器人用手/身体把球夹到球拍附近、反复吃接近奖励或辅助触发击球的 reward hacking。

### 实现记录

- Hit 任务将 `paddle_hit_event` 权重从 `100.0` 提高到 `2000.0`；在当前 `dt=0.02`
  下单次合法击球约为 `+40`。
- Hit dense shaping 降低为 `approach_ball=5.0`、`paddle_towards_ball=2.0`，Return
  仍保留自己的 `paddle_hit_event=25.0`，避免 Return 目标退化为只追求首次碰球。
- 新增 `robot_ball_contact` sensor：机器人碰撞几何为 primary，排除
  `pingpong_paddle_collision`，球为 secondary，`history_length=4`。
- `PingpongRallyState` 增加 `FAULT_ILLEGAL_BODY_BALL_CONTACT`，任意非球拍
  robot-ball 接触会触发 fault；同一步身体和球拍都碰球时 fault 优先，不计合法 hit。
- 新增 `robot_ball_contact` penalty，权重 `-50.0`，不新增单独 metric；主要诊断仍看
  `fault_count`、fault reason 和视频。

### 验证

- `UV_CACHE_DIR=/tmp/uv-cache FORCE_CPU=1 uv run pytest tests/test_pingpong_state.py tests/test_pingpong_task.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/mjlab/tasks/pingpong tests/test_pingpong_state.py tests/test_pingpong_task.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ty check src/mjlab/tasks/pingpong`

## 2026-06-24 - Pingpong Cross Stabilization

### 目标

针对 Cross warm-start 后击球阶段身体不受控的问题，保存当前可视化视频，并调低探索/动作幅度、
增强站稳与动作平滑约束后继续训练。

### 实现记录

- 使用 `model_3500.pt` 保存 10 秒视频：
  `logs/rsl_rl/g1_pingpong_latent_cross/pingpong_cross_from_v3_collision_4500_2026-06-24_17-10-59/videos/play/rl-video-step-0.mp4`。
- `TennisLatentOnPolicyRunnerCfg` 新增 `reset_actor_std`，加载 checkpoint 时可重置
  actor Gaussian std；当前 relaxed 版默认设为 `0.8`。
- Cross PPO 当前调整：`entropy_coef=0.002`、`learning_rate=5e-4`、`desired_kl=0.01`、
  `clip_actions=2.5`、`reset_actor_std=0.8`。
- Cross reward/termination 调整：提高 latent/low-level action rate、joint acc/torque、
  fall/upright 惩罚；加入 `flat_orientation_l2`；第一版将 post-hit dense shaping
  降为 `post_hit_x_progress=20`、`post_hit_ball_velocity_direction=10`，当前 relaxed
  版恢复为 `post_hit_x_progress=40`、`post_hit_ball_velocity_direction=20`；收紧
  `bad_orientation=55deg`、`root_height=0.55`。
- 当前 relaxed 正则项：`latent_action_rate_l2=-0.01`、`low_level_action_rate_l2=-0.02`、
  `joint_acc_l2=-5e-6`、`joint_torques_l2=-5e-5`。
- 旧 Cross 训练停止后落出 `model_4000.pt`；稳定版训练已按用户要求挂到 GPU
  `[4,6]`、`16384` env，session 为 `pingpang_cross_stable_16384_gpu4_6`，输出目录：
  `logs/rsl_rl/g1_pingpong_latent_cross/pingpong_cross_stable_from4000_16384env_gpu4_6_2026-06-24_21-21-58`。

### 验证

- `UV_CACHE_DIR=/tmp/uv-cache FORCE_CPU=1 uv run pytest tests/test_pingpong_task.py tests/test_pingpong_state.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/mjlab/tasks/pingpong src/mjlab/tasks/tennis/rl tests/test_pingpong_task.py tests/test_pingpong_state.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ty check src/mjlab/tasks/pingpong src/mjlab/tasks/tennis/rl`

## 2026-06-24 - Pingpong Paddle Handle Collision

### 目标

为 pingpong 球拍补充一个圆柱形拍柄碰撞体，使拍柄参与球桌/球体物理碰撞，但不参与合法击球得分。

### 实现记录

- 在 G1 pingpong spec wrapper 中新增 `pingpong_paddle_handle_collision`，位于
  `pingpong_paddle_collision` 下方，沿视觉拍柄方向放置。
- 拍柄参数：半径 `0.018m`，半长 `0.09m`；编译后 `contype=1`、
  `conaffinity=1`。
- 拍柄是纯碰撞代理，设置为 `group=3` 且 alpha 为 `0`，避免在正常可视化中显示。
- `paddle_ball_contact` 仍只匹配 `pingpong_paddle_collision`；拍柄未从
  `robot_ball_contact` 排除，因此拍柄碰球不会计为合法 hit，会按非拍面机器人碰球处理。
- 生成正式模型确认图：
  `contact_test/pingpong_paddle_formal_handle_collision_2026-06-24.png`。

### 验证

- `UV_CACHE_DIR=/tmp/uv-cache FORCE_CPU=1 uv run pytest tests/test_pingpong_task.py tests/test_pingpong_state.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/mjlab/tasks/pingpong/config/g1/env_cfgs.py src/mjlab/tasks/pingpong/pingpong_env_cfg.py tests/test_pingpong_task.py tests/test_pingpong_state.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ty check src/mjlab/tasks/pingpong/config/g1/env_cfgs.py src/mjlab/tasks/pingpong/pingpong_env_cfg.py tests/test_pingpong_task.py tests/test_pingpong_state.py`

## 2026-06-24 - Pingpong Cross Task

### 目标

新增正式的 `Mjlab-Pingpong-Cross-Unitree-G1` 任务，目标是让机器人完成一次合法过网回球：
己方桌面弹起后击球，球过网，并在对方桌面首次有效落点。

### 实现记录

- 将现有 Return 的合法回球语义提炼为 `make_pingpong_latent_cross_env_cfg()`；
  `make_pingpong_latent_return_env_cfg()` 保留为兼容别名。
- 新增 G1 env wrapper、PPO runner cfg 和 task 注册：
  `experiment_name="g1_pingpong_latent_cross"`，
  `run_name="pingpong_cross_from_hit"`。
- Cross 默认不硬编码本地 Hit checkpoint；训练时通过 CLI 传入当前最好 Hit checkpoint。
- Reward/termination 沿用已验证的 Return 语义：`paddle_hit_event=25` 只作为阶段奖励，
  `crossed_net_event=500`、`opponent_table_bounce_event=1000` 作为主要回球成功路径，
  `legal_return_success` 作为 curriculum success term。

### 验证

- `UV_CACHE_DIR=/tmp/uv-cache FORCE_CPU=1 uv run pytest tests/test_pingpong_state.py tests/test_pingpong_task.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/mjlab/tasks/pingpong tests/test_pingpong_state.py tests/test_pingpong_task.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ty check src/mjlab/tasks/pingpong`

## 2026-06-25 - Pingpong Hit Regularization Curriculum

### 目标

在 Hit 任务中先完成现有落点课程，再根据 `first_paddle_hit` 成功率逐步增强
动作平滑和稳定性正则，最终到达后续 Cross 可逐步收紧的强正则目标。

### 实现记录

- 新增 `success_reward_weight_curriculum`：基于 done episode 的成功率窗口推进，
  可同时更新多个 reward term 的 `weight`。
- `CurriculumManager` 新增只读 `get_term_state()`，供后续课程读取前置课程状态。
- Hit 新增 `action_regularization` 课程，等待 `ball_target_region.stage == 5` 后开始统计；
  成功率阈值 `0.80`，窗口 `50`。
- 正则课程覆盖 `latent_action_rate_l2`、`joint_torques_l2`、`joint_acc_l2`、
  `fall_penalty`、`flat_orientation_l2`；`low_level_action_rate_l2` 保持 `-0.02`。
- Cross/Return 不启用该课程；当前 Cross 默认会先用更宽松的 return-first 正则学会回球。

### 验证

- `UV_CACHE_DIR=/tmp/uv-cache FORCE_CPU=1 uv run pytest tests/test_envs_curriculums.py tests/test_pingpong_task.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/mjlab/envs/mdp/curriculums.py src/mjlab/managers/curriculum_manager.py src/mjlab/tasks/pingpong tests/test_envs_curriculums.py tests/test_pingpong_task.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ty check src/mjlab/envs/mdp src/mjlab/managers src/mjlab/tasks/pingpong`

## 2026-06-25 - Pingpong Cross Return-First 松绑

### 目标

停止当前保守的 Cross 训练，将 Cross 默认配置改为先学习稳定合法回球，再在后续阶段考虑
action-rate / torque / acceleration 等正则收紧。

### 实现记录

- 停止 tmux session `pingpang_cross_from_hit_v3_16384_gpu4_6`。
- Cross/Return 保留 `robot_ball_contact` fault、`robot_table_contact` penalty、
  `legal_return_success`、`crossed_net_event=500`、`opponent_table_bounce_event=1000`、
  `post_hit_x_progress=40`、`post_hit_ball_velocity_direction=20`。
- 新增 `CROSS_LOOSE_REGULARIZATION_WEIGHTS` 作为 Cross 默认宽松正则：
  `latent_action_rate_l2=-0.005`、`low_level_action_rate_l2=-0.01`、
  `joint_torques_l2=-2e-5`、`joint_acc_l2=-2e-6`、`fall_penalty=-300`、
  `flat_orientation_l2=-0.5`。

### 验证

- `UV_CACHE_DIR=/tmp/uv-cache FORCE_CPU=1 uv run pytest tests/test_envs_curriculums.py tests/test_pingpong_task.py tests/test_pingpong_state.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/mjlab/envs/mdp/curriculums.py src/mjlab/managers/curriculum_manager.py src/mjlab/tasks/pingpong src/mjlab/tasks/tennis/rl tests/test_envs_curriculums.py tests/test_pingpong_task.py tests/test_pingpong_state.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ty check src/mjlab/envs/mdp src/mjlab/managers src/mjlab/tasks/pingpong src/mjlab/tasks/tennis/rl`

## 2026-06-25 - Pingpong Cross Dense Reward Retune

### 目标

根据当前 Cross loose run 的 reward 分布，降低压制挥拍和偶发身体碰球的负项，
同时增强击球后过网方向的 dense shaping。

### 实现记录

- Cross `joint_acc_l2` 从 `-2e-6` 降到 `-1e-6`，减少击球后加速动作惩罚。
- Cross/Return 单独将 `robot_ball_contact` reward penalty 从 `-50` 降到 `-25`；
  Hit 仍保持 `-50`，身体碰球 fault 终止也保留。
- Cross `post_hit_x_progress` 从 `40` 提到 `80`。
- Cross `post_hit_ball_velocity_direction` 从 `20` 提到 `60`。

### 验证

- `UV_CACHE_DIR=/tmp/uv-cache FORCE_CPU=1 uv run pytest tests/test_pingpong_task.py tests/test_pingpong_state.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/mjlab/tasks/pingpong tests/test_pingpong_task.py tests/test_pingpong_state.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ty check src/mjlab/tasks/pingpong`

## 2026-06-29 20:32 - Pingpong Cross Hit-to-Return Diagnostics

### 目标

定位 Cross 卡在 hit-to-return funnel 的原因，补充可观测诊断，增加可开关的
strike-quality reward ablation，并建立 14h 长训 watcher。

### 计划

- [x] 复查 Cross 状态机、reward、metrics、G1 注册和历史 run 指标。
- [x] 记录合法 hit 后的球速、方向、预测过网高度、预测落点和球拍侧质量。
- [x] 暴露 fault reason breakdown 与 robot-ball contact episode metric。
- [x] 新增 diagnostics-only、strike-quality、strike-quality+energy-relax 三个 ablation task。
- [x] 新增 watcher，按 84 分钟 observation window 生成 JSON/Markdown 决策报告。

### 实现记录

- `PingpongRallyState` 在首次合法 paddle hit 时缓存 post-hit ball velocity、
  speed、toward-opponent ratio、predicted net clearance、predicted opponent-table
  landing 和 paddle speed / normal alignment / normal velocity。
- Metrics 新增 `fault_reason/*`、`robot_ball_contact_count`、`hit/post_*`、
  `hit/pred_*`、`hit/paddle_*`；旧 Cross reward 默认不自动启用新增 strike reward。
- 新增 strike-quality reward：`strike_outgoing_ball_velocity`、
  `strike_pred_net_clearance`、`strike_pred_landing_inside`、
  `strike_post_hit_speed`。
- 新增 `pre_hit_action_rate_l2` 与 `pre_hit_low_level_action_rate_l2`，用于
  `Mjlab-Pingpong-Cross-StrikeQualityEnergyRelax-Unitree-G1` 在合法 hit 后的
  return-flight 阶段放松 action-rate penalty，而不是全局放飞动作。
- 新增 `tools/watch_pingpong_cross_training.py`，支持 `--logdir`、`--pid`、
  `--eta-hours`、`--observe-interval-minutes`、`--kill-on-stop`、`--dry-run`、
  `--once` 和 `--report-dir`。

### 问题判断

历史 dense-retune run 的证据仍支持 strike-quality bottleneck：自方落台稳定，
paddle 可以碰到约一半来球，但过网、对方落台和合法回球均为 0；同时
`post_hit_x_progress` 与 `post_hit_ball_velocity_direction` 很低。

### 验证

- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check ...`
- `UV_CACHE_DIR=/tmp/uv-cache FORCE_CPU=1 uv run pytest tests/test_pingpong_state.py tests/test_pingpong_task.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ty check src/mjlab/tasks/pingpong tests/test_pingpong_state.py tests/test_pingpong_task.py tools/watch_pingpong_cross_training.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python tools/watch_pingpong_cross_training.py --help`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python tools/watch_pingpong_cross_training.py --logdir /data0/universe/home_moved/mjlab_latent/logs/rsl_rl/g1_pingpong_latent_cross/pingpong_cross_dense_retune_from_hit_v3_collision4500_16384env_gpu4_6_2026-06-25_14-26-38 --eta-hours 14 --observe-interval-minutes 84 --report-dir /tmp/pingpong_cross_watch_dryrun3 --dry-run --once`

### Pre-commit review follow-up

- Paddle-side hit quality originally assumed the paddle site local `+x` axis.
  Review changed it to the `pingpong_paddle_collision` geom local `+z` normal,
  matching the thin-cylinder collision proxy used for legal hits.
- Legal-hit diagnostics no longer call `bool(torch.any(...))`; vectorized masked
  updates avoid a CPU/GPU sync in the hot path.
- `pred_net_clearance` and predicted landing now reject or cap very long
  ballistic horizons, avoiding misleading huge negative clearance from nearly
  stationary/away-from-net post-hit velocity.
- Dense reward terms that share `_state_params()` now accept the additional
  `paddle_geom_cfg` argument, avoiding config construction regressions.
- Final validation before launch passed:
  `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format ...`,
  `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check ...`,
  `UV_CACHE_DIR=/tmp/uv-cache FORCE_CPU=1 uv run pytest tests/test_pingpong_state.py tests/test_pingpong_task.py -q`,
  and `UV_CACHE_DIR=/tmp/uv-cache uv run ty check src/mjlab/tasks/pingpong tests/test_pingpong_state.py tests/test_pingpong_task.py tools/watch_pingpong_cross_training.py`.

### Diag-only long run launch

- First attempt with `30720` envs on GPU `[0,1]` hit Warp CUDA graph OOM.
- Second attempt with `16384` envs exposed the `paddle_geom_cfg` reward
  signature bug; fixed before relaunch.
- Third attempt exposed the long-horizon `pred_net_clearance` diagnostic issue;
  fixed before relaunch.
- Active run:
  `Mjlab-Pingpong-Cross-Diag-Unitree-G1`,
  tmux `pingpong_cross_diag_20260629_211353_16384`,
  pane PID `3180955`,
  `16384` envs on GPU `[0,1]`.
- Logdir:
  `/data0/universe/home_moved/mjlab_latent/logs/rsl_rl/g1_pingpong_latent_cross_diag/pingpong_cross_diag_only_from_hit_16384env_gpu0_1_20260629_211353_2026-06-29_21-15-42`.
- Launch record:
  `/data0/universe/home_moved/mjlab_latent/logs/rsl_rl/_codex_launch_records/pingpong_cross_diag_20260629_211353_16384`.
- Watcher `--once` succeeded with `decision=continue`, no missing metrics, and
  wrote
  `/data0/universe/home_moved/mjlab_latent/logs/rsl_rl/g1_pingpong_latent_cross_diag/pingpong_cross_diag_only_from_hit_16384env_gpu0_1_20260629_211353_2026-06-29_21-15-42/watch_reports/20260629_212039_pingpong_cross_watch.md`.
- Persistent watcher tmux:
  `pingpong_cross_diag_watch_20260629_211353_16384`; it observes every
  `84` minutes and writes reports under the run's `watch_reports/`.

### 下一步

优先跑 `Mjlab-Pingpong-Cross-Diag-Unitree-G1` 确认新 metrics；如果
`hit/post_vx_toward_opponent_ratio`、`hit/pred_net_clearance` 和
`hit/pred_landing_inside_opponent_table` 仍长期为 0，再比较 strike-quality reward
与 hit-window energy-relax ablation。

## 2026-07-02 17:30 - Pingpong Cross hit-recovery restart

### 目标

针对 `pingpong_cross_impact_collision_update_15360...` 训练中
`paddle_hit_count` 下降、`robot_ball_contact_count` 和
`fault_reason/body_ball` 上升的问题，先把训练目标重新拉回合法拍面命中。

### 参数调整

- `paddle_hit_event.weight`: `40.0 -> 200.0`
- `CROSS_ROBOT_BALL_CONTACT_WEIGHT`: `-25.0 -> -75.0`
- `CROSS_POST_HIT_X_PROGRESS_WEIGHT`: `120.0 -> 60.0`
- `Mjlab-Pingpong-Cross-Impact-Unitree-G1` PPO:
  `entropy_coef=0.001`, `reset_actor_std=0.6`

### 运行处理

- 已停止旧 tmux：
  `pingpong_cross_impact_collision_update_15360_20260702_144131`
- 用户确认不要从旧 impact ckpt 接训；新 run 直接从 Hit 最好 ckpt warm start：
  `/data0/universe/home_moved/mjlab_latent/logs/rsl_rl/g1_pingpong_latent_hit/v3_collision_2026-06-24_15-53-52/model_4500.pt`
- 新 tmux：
  `pingpong_cross_impact_from_hit_15360_20260702_172724`
- 新 run：
  `/data0/universe/home_moved/mjlab_latent/logs/rsl_rl/g1_pingpong_latent_cross_impact/pingpong_cross_impact_from_hit_15360pergpu_gpu4_7_20260702_172724_2026-07-02_17-28-34`
- Launch record：
  `/data0/universe/home_moved/mjlab_latent/logs/rsl_rl/_codex_launch_records/pingpong_cross_impact_from_hit_15360_20260702_172724/COMMAND.md`
- GPU: `CUDA_VISIBLE_DEVICES=4,5,6,7`, `15360` env per visible GPU,
  `max_iterations=2000`, `reset_resume_progress=True`

### 验证

- `UV_CACHE_DIR=/tmp/uv-cache FORCE_CPU=1 uv run pytest tests/test_pingpong_task.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/mjlab/tasks/pingpong/pingpong_env_cfg.py src/mjlab/tasks/pingpong/config/g1/rl_cfg.py tests/test_pingpong_task.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ty check src/mjlab/tasks/pingpong/pingpong_env_cfg.py src/mjlab/tasks/pingpong/config/g1/rl_cfg.py tests/test_pingpong_task.py`
- Resolved agent config confirms `load_checkpoint_file=model_4500.pt`,
  `entropy_coef=0.001`, `reset_actor_std=0.6`.

### 观察重点

先看 `paddle_hit_count` 是否回升，同时 `robot_ball_contact_count` /
`fault_reason/body_ball` 是否下降；若 hit 回升但 `legal_return_count` 长期不动，
再恢复或重做更直接的 impact-window paddle behavior rewards。

## 2026-07-06 01:30 - Table-tennis distill play video

### 目标

用 `play-distill.py` 直接加载 v1 table-tennis decoder，录制 `test_001`
motion tracking 的 mp4，便于检查低层 decoder 的动作跟踪质量。先前误录过一版
v2 `model_1750.pt`，但本次应以 v1 `model_30000.pt` 视频为准。

### 改动

- `play-distill.py` 新增非交互视频录制参数：
  `--video`、`--video-folder`、`--video-length`、`--video-width`、
  `--video-height` 和 `--video-name-prefix`。
- 录制路径复用项目 `VideoRecorder`，普通 native/viser viewer 播放路径不变。

### 录制对象

- Checkpoint:
  `/data0/universe/home_moved/mjlab_latent/logs/rsl_rl/g1_distillation_table_tennis/table_tennis_distill_v1_46080env_from_tracking18000_2026-07-03_10-14-26/model_30000.pt`
- Motion:
  `artifacts/table_tennis/test_001/motion.npz`
- Latent:
  posterior expected z，`state_dim=93`，`target_dim=67`，`latent_dim=16`
- Render:
  `MUJOCO_GL=egl`，`CUDA_VISIBLE_DEVICES=7`，`960x540`，`600` frames

### 产物

- Actual video:
  `outputs/play_distill_table_tennis_v1_model30000_test001_posterior/table_tennis_distill_v1_model30000_test001_posterior-step-0.mp4`
- Convenience symlink:
  `outputs/table_tennis_distill_v1_model30000_test001_play_distill.mp4`
- Teacher tracking reference video:
  `outputs/play_tracking_table_tennis_teacher_model18000_test001/videos/play/rl-video-step-0.mp4`
- Teacher convenience symlink:
  `outputs/table_tennis_teacher_tracking_model18000_test001.mp4`

### 验证

- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check play-distill.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ty check play-distill.py`
- `imageio` metadata confirms codec `h264`, pixel format `yuv420p`,
  `50.0` FPS, duration `12.0s`, size `960x540`, and non-empty first frame.

## 2026-07-07 09:31 - Table-tennis v2 distill comparison video

### 目标

对比 v2 `model_30000.pt` 在 `test_001` 上的 tracking teacher、posterior
decoder 和 prior decoder 表现。

### 录制对象

- Decoder checkpoint:
  `logs/rsl_rl/g1_distillation_table_tennis/table_tennis_distill_v2_46080env_kl_aligned_from_tracking18000_gpu1_2026-07-05_23-43-32/model_30000.pt`
- Teacher reference:
  `outputs/table_tennis_teacher_tracking_model18000_test001.mp4`
- Motion:
  `artifacts/table_tennis/test_001/motion.npz`
- Render:
  `MUJOCO_GL=egl`，`CUDA_VISIBLE_DEVICES=1`，`960x540`，`600` frames

### 产物

- V2 posterior:
  `outputs/table_tennis_distill_v2_model30000_test001_posterior.mp4`
- V2 prior:
  `outputs/table_tennis_distill_v2_model30000_test001_prior.mp4`
- Three-panel comparison:
  `outputs/table_tennis_v2_model30000_test001_teacher_posterior_prior_compare.mp4`

### 验证

- `imageio` metadata confirms the three-panel comparison is `h264`,
  `yuv420p`, `50.0` FPS, duration `12.0s`, size `2880x592`, and has a
  non-empty first frame.

## 2026-07-07 09:44 - Pingpong Hit with table-tennis v2 decoder

### 目标

虽然 table-tennis v2 decoder 的 tracking 视频仍不完美，但将其接入
`Mjlab-Pingpong-Hit-Unitree-G1` 做高层 Hit RL 试验，观察是否能学到主动拍面命中。

### 启动配置

- Tmux:
  `pingpong_hit_tt_decoder_v2_gpu1_20260707`
- GPU:
  `CUDA_VISIBLE_DEVICES=1`
- Task:
  `Mjlab-Pingpong-Hit-Unitree-G1`
- Num envs:
  `15360`
- Experiment:
  `g1_pingpong_latent_hit_table_tennis_decoder`
- Run:
  `pingpong_hit_tt_decoder_v2_model30000_15360env_gpu1_20260707_2026-07-07_09-44-02`
- Decoder:
  `logs/rsl_rl/g1_distillation_table_tennis/table_tennis_distill_v2_46080env_kl_aligned_from_tracking18000_gpu1_2026-07-05_23-43-32/model_30000.pt`

### 验证

- 启动前 1-env CPU smoke 成功加载 v2 decoder 并 step 一次，未出现
  decoder state/action mismatch。
- Resolved config 确认：
  `env.scene.num_envs=15360`，`agent.max_iterations=30000`，
  `agent.upload_model=false`，且
  `env.actions.latent_joint_pos.decoder_checkpoint` 指向 v2 `model_30000.pt`。
- run 目录写入：
  `logs/rsl_rl/g1_pingpong_latent_hit_table_tennis_decoder/pingpong_hit_tt_decoder_v2_model30000_15360env_gpu1_20260707_2026-07-07_09-44-02/COMMAND.md`

### 早期观察

- 约前几轮：`paddle_hit_count` 非零并升到约 `0.16`，说明有命中信号。
- `robot_ball_contact_count≈0.69`、`fault_reason/body_ball≈0.53` 仍然高，
  需要继续观察是否能从身体挡球/蹭球局部行为转向干净拍面击球。
- `fall_penalty=0`、bad orientation/root height 早期为 0，站立稳定性暂时可接受。

## 2026-07-06 20:50 - Pingpong PACE 12x1024 GPU3 restart

### 目标

将 PACE baseline 从疑似挤爆的 `16*1024` / `14*1024` env run 降到
`12*1024` env，继续在 host GPU3 上训练，并清理失败 run 目录。

### 处理

- 失败 session `pingpong_pace_16384_gpu3_20260706` 已不在 `tmux ls` 中。
- 已清理失败 run 目录：
  `logs/rsl_rl/g1_pingpong_pace/pingpong_pace_scratch_16384env_gpu3_2026-07-06_20-39-11`。
- 失败 session `pingpong_pace_14336_gpu3_20260706` 也已不在 `tmux ls` 中。
- 已清理失败 run 目录：
  `logs/rsl_rl/g1_pingpong_pace/pingpong_pace_scratch_14336env_gpu3_2026-07-06_20-44-37`。
- 新 session：
  `pingpong_pace_12288_gpu3_20260706`
- 新 run：
  `logs/rsl_rl/g1_pingpong_pace/pingpong_pace_scratch_12288env_gpu3_2026-07-06_20-49-36`
- W&B run id:
  `ysevx7op`

### 启动命令

```sh
CUDA_VISIBLE_DEVICES=3 UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig \
WANDB_CACHE_DIR=/tmp/wandb-cache WANDB_CONFIG_DIR=/tmp/wandb-config \
XDG_CACHE_HOME=/tmp/xdg-cache MJLAB_WARP_QUIET=1 \
uv run train Mjlab-Pingpong-PACE-Unitree-G1 \
  --env.scene.num-envs 12288 \
  --gpu-ids "[0]" \
  --agent.experiment-name g1_pingpong_pace \
  --agent.run-name pingpong_pace_scratch_12288env_gpu3 \
  --agent.upload-model False
```

### 初始观察

- `12*1024` env 第 0 个 iteration 已完成，说明不只是初始化成功。
- GPU3 显存约 `20465 / 24564 MiB`，utilization `97%`。
- Iteration 0: `Steps per second=41294`，`predictor_mse=0.1185`，
  `Mean reward=-21.22`。
- Scratch 初期 `paddle_hit_count=0`、`legal_return_count=0` 属预期现象；
  后续主要看 contact、crossed net、opponent bounce 和 legal return 是否抬头。

## 2026-07-06 22:53 - PACE reference audit in /tmp

### 目标

按用户要求把 PACE 官方参考实现放到 `/tmp`，并记录它对当前
`Mjlab-Pingpong-PACE-Unitree-G1` 不稳定问题的解释价值。

### 参考代码

- Clone path: `/tmp/PACE-ICRA2026`
- Source: `git@github.com:purdue-tracelab/PACE-ICRA2026.git`
- Clone command:

```sh
git clone --depth 1 https://github.com/purdue-tracelab/PACE-ICRA2026.git /tmp/PACE-ICRA2026
```

### 参考实现要点

- PACE 的 table-tennis 训练入口是 `t1_tt`，README 推荐
  `--num_envs=4096 --predictor`。它不是 latent low-level，而是直接
  joint-position policy。
- 动作处理在 `legged_lab/envs/base/tt_env.py`：
  `processed_actions = clip(actions, -100, 100) * action_scale + default_joint_pos`。
  其中 T1 配置使用全局 `action_scale=0.25`，`clip_actions=100.0`。
- T1 table-tennis 环境控制 21 个关节，`sim.dt=0.002`、
  `decimation=10`，即 50 Hz。
- PPO 配置使用 `num_steps_per_env=24`、`init_noise_std=1.0`、
  scalar std、`entropy_coef=0.006`、`gamma=0.95`、
  `empirical_normalization=False`。
- Actor observation 用 learned ball prediction；critic observation 用
  physics future ball pose、paddle touch point、future robot target delta、
  future hit time 和 contact/phase flags。
- Predictor runner 每步记录 ball position，前 `20` 个 PPO iteration 用
  physics future pose 做监督训练，训练后每步调用 env
  `update_prediction(preds)`，checkpoint 中保存 `pred_state_dict`、
  `pred_optimizer_state_dict` 和 `pred_cfg`。
- 核心击球 reward 权重和我们迁移的一致：contact `150`、future EE `2`、
  future body `5`、future base velocity `5`、future landing `60`、
  future pass net `100`、table success `100`。
- PACE 原版还有一组 v1 迁移时没有完整带过来的稳定项：`fly`、
  `hit_unstable_support`、左右脚姿态、脚底受力、脚滑、脚绊、
  两脚过近、paddle head 过近身体等。

### 对当前问题的判断

- 失败的 `pingpong_pace_scratch_12288env_gpu3` 不是显存 OOM，而是在
  iteration 515 左右出现 actor observation NaN。
- 该 run 的 `model_500.pt` policy/predictor 权重 finite，但 policy std 已从
  `1.0` 长到约 `3.65`，同时 `action_rate_l2`、`joint_pos_limits`、
  `root_height` 和 `bad_orientation` 明显恶化。
- 因此风险点更像是 PACE 控制 envelope 没有完全迁移：原版 T1 是 21 关节、
  全局 `0.25` action scale 和完整稳定 reward；我们当前 G1 direct-PACE 是
  29 关节、按 G1 乒乓球拍模型 action scale，部分关节 scale 约 `0.55`，
  再配合 `clip_actions=100` 时实际几乎没有动作裁剪。

### 下一步建议

在重新跑 direct-PACE 前，优先对齐控制 envelope，而不是先动球路或
Cross-StrikeQuality：

- 降低 G1 direct-PACE 的动作尺度，或给 direct-PACE 增加实际有效的动作裁剪。
- 降低探索增长风险，例如收紧 initial/std cap 或 entropy。
- 若目标是更完整复刻 PACE，再补齐上面列出的稳定项，并单独测试不影响普通
  Cross / StrikeQuality / latent decoder。

## 2026-07-06 23:06 - G1 PACE baseline stabilization

### 目标

按计划先稳定 `Mjlab-Pingpong-PACE-Unitree-G1` direct joint-position baseline，
避免 policy std 放大后把 G1 推到异常物理状态。普通 latent Cross /
StrikeQuality、decoder checkpoint、球路分布和持拍模型保持不变。

### 代码改动

- 新增 PACE-only G1 action scale：
  `G1_PACE_ACTION_SCALE = min(G1_W_RACKET_ACTION_SCALE, 0.25)`，只用于
  `unitree_g1_pingpong_pace_env_cfg()`。
- PACE PPO 改为保守探索：
  `clip_actions=2.5`、`init_std=0.6`、`entropy_coef=0.002`、
  `gamma=0.95`，actor/critic hidden dims 改为 `(512, 512, 128)`。
- Predictor 配置保持不变。
- 增加不依赖新 foot contact sensor 的 PACE 稳定 reward：
  `pace_fly_height=-2.5`、`pace_hit_unstable_support_height=-10.0`、
  `pace_feet_orientation_left/right=-4.0`、`pace_feet_too_near=-1.5`、
  `pace_feet_really_too_near=-10.0`。
- 暂不加入 PACE 原版 feet-force / stumble 接触力项；这需要专用足部
  contact sensor，留到 v2。

### 验证计划

- 先跑相关 pytest 和 ty check。
- 再跑 CPU smoke train：
  `FORCE_CPU=1 UV_CACHE_DIR=/tmp/uv-cache uv run train Mjlab-Pingpong-PACE-Unitree-G1 --agent.max-iterations 1 --env.scene.num-envs 2 --agent.upload-model False`
- CPU 验证通过后，在 GPU3 启动 4096-env canary，确认无 NaN、`Policy/mean_std`
  不再快速涨到 `3+`，并观察 action-rate / joint-limit / root-height /
  bad-orientation 是否不再同步恶化。

### 验证结果

- `FORCE_CPU=1 UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pingpong_task.py tests/test_pingpong_state.py -q`
  通过：`24 passed`。
- `UV_CACHE_DIR=/tmp/uv-cache uv run ty check src/mjlab/tasks/pingpong tests/test_pingpong_task.py tests/test_pingpong_state.py`
  通过。
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/mjlab/tasks/pingpong tests/test_pingpong_task.py tests/test_pingpong_state.py`
  通过。
- CPU smoke 首次只设置 `FORCE_CPU=1` 时失败在默认 `gpu_ids=[0]` 对空 GPU 列表
  取下标；重跑时显式加 `--gpu-ids "[]"` 通过。
- CPU smoke passing command:

```sh
FORCE_CPU=1 WANDB_MODE=offline UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig \
uv run train Mjlab-Pingpong-PACE-Unitree-G1 \
  --gpu-ids "[]" \
  --agent.max-iterations 1 \
  --env.scene.num-envs 2 \
  --agent.upload-model False
```

### GPU canary

- Session:
  `pingpong_pace_stable_canary_4096_gpu3_20260706`
- Run:
  `logs/rsl_rl/g1_pingpong_pace/pingpong_pace_stable_canary_4096env_gpu3_2026-07-06_23-10-26`
- W&B run id:
  `h9859s2u`
- Launch record:
  `logs/rsl_rl/_codex_launch_records/pingpong_pace_stable_canary_4096_gpu3_20260706/COMMAND.md`
- Initial GPU3 memory after startup:
  about `6133 / 24564 MiB`.
- Iteration 15 early check:
  `Mean action std=0.61`, `Episode_Termination/nan_detection=0.0`,
  `bad_orientation=0.0`, `root_height=0.0417`, and no early actor-observation NaN.
- Final canary result:
  this run was scratch, not resumed from a failed checkpoint. It later failed at
  iteration 592 with actor observation NaN. Near failure, `Mean action std` was
  about `0.77`, `action_rate_l2` about `-0.2941`, `joint_pos_limits` about
  `-0.0232`, `bad_orientation` about `0.8750`, and `root_height` about
  `3.2083`. This shows the remaining failure is still the direct-control task /
  environment stability, not checkpoint inheritance.

## 2026-07-06 23:50 - G1 PACE formal scratch run

### 目标

按用户要求正式用 tmux 直接 scratch 训练，不从任何失败 PACE checkpoint 继续。

### 启动配置

- Session:
  `pingpong_pace_stable_scratch_12288_gpu3_20260706`
- Run:
  `logs/rsl_rl/g1_pingpong_pace/pingpong_pace_stable_scratch_12288env_gpu3_2026-07-06_23-50-08`
- W&B run id:
  `apqf2mva`
- Launch record:
  `logs/rsl_rl/_codex_launch_records/pingpong_pace_stable_scratch_12288_gpu3_20260706/COMMAND.md`
- GPU:
  host GPU3, `12288` envs, peak observed memory about `15029 / 24564 MiB`
- Checkpoint policy:
  `--agent.resume False`, no `load_checkpoint_file`, no `load_run`,
  no `load_checkpoint`

### 结果

- Iteration 5:
  `Mean action std=0.60`, `nan_detection=0.0`, `bad_orientation=0.0`,
  `root_height=0.0417`。
- Iteration 6:
  still `Mean action std=0.60`, `nan_detection=0.0`, `bad_orientation=0.0`,
  `root_height=0.0833` in logged episode stats.
- Immediately after iteration 6, the run exited with:
  `ValueError: The observation group 'actor' returned by the environment contains NaN values.`

### 判断

这次正式 run 已经证明不是在失败 ckpt 上续训导致的问题。相同代码从 scratch
在 `12288` env 下会很快触发 rare actor-observation NaN；`4096` env 下能跑到
约 `592` iteration 后才触发。下一步不应重复启动同样配置的正式训练，而应先给
PACE runner/env 增加 NaN source dump，定位是 `joint_pos`、`joint_vel`、
`ball_pos`、`robot_pos`、`ball_prediction`、`rel_target_base_xy` 还是
`heading` 产生 NaN，并记录对应 env id / reset state / qpos / qvel。
- Canary failed at iteration 592 with actor observation NaN. This is an
  improvement over the old run's std explosion (`Mean action std` stayed around
  `0.77` instead of `3+`), but not acceptable for formal training. Near failure,
  `Episode_Termination/bad_orientation` was around `0.875` and
  `Episode_Termination/root_height` around `3.2083`, so the remaining issue is
  direct-control instability/fall states rather than predictor NaNs or GPU OOM.

### Follow-up stabilization before formal 12288-env training

- Tighten PACE-only direct-control envelope further:
  `G1_PACE_ACTION_SCALE` cap `0.25 -> 0.18`,
  `clip_actions 2.5 -> 2.0`,
  actor `init_std 0.6 -> 0.45`,
  `entropy_coef 0.002 -> 0.001`.
- Make PACE terminate unstable body states earlier:
  `bad_orientation.limit_angle=45deg`,
  `root_height.minimum_height=0.65`.

## 2026-07-07 09:35 - G1 PACE actor NaN root-cause diagnosis

### 目标

定位 PACE scratch run 的 actor observation NaN 原因，并保留原始报错证据。

### 原始报错

两条 scratch run 的原始异常一致：

```text
ValueError: The observation group 'actor' returned by the environment contains NaN values. This usually indicates a bug in the environment's step() or reset() function.
```

失败位置：

- `src/mjlab/tasks/pingpong/rl/runner.py` 调用
  `check_nan(obs, rewards, dones)`。
- `rsl_rl/utils/utils.py:279` 抛出上述 `ValueError`。

### 诊断改动

- 在 `PingpongPaceOnPolicyRunner` 中增加 PACE-only NaN diagnostic dump：
  触发 NaN 时在 run 目录下写
  `diagnostics/pace_nan_iter*_step*.json`，记录 group/term 级别
  NaN/Inf 计数、bad env id、robot/ball root state、last action 和 PACE
  predictor state。
- 保持原始 `check_nan` 异常行为不变，只是在抛错前写诊断文件。

### 复现与证据

- 诊断 run：
  `logs/rsl_rl/g1_pingpong_pace/pingpong_pace_nan_diag_12288env_gpu3_2026-07-07_09-25-33`
- 诊断文件：
  `diagnostics/pace_nan_iter000001_step013.json`
- bad env:
  `9563`
- actor group:
  only `rel_target_base_xy` has NaN, `nan_count=2`。
- critic group:
  `future_ball_pose` has `nan_count=3` and
  `robot_future_delta` has `nan_count=2`。
- `actions`、`joint_pos`、`joint_vel`、`base_ang_vel`、`ball_pos`、
  `robot_pos`、`ball_prediction` all remained finite.
- bad env snapshot showed `episode_length_buf=0` and `last_action=0`,
  i.e. the environment had just been auto-reset; post-reset robot/ball state
  was finite.

### 根因

`PingpongPacePredictionState.update()` cached by global
`common_step_counter`. During `ManagerBasedRlEnv.step()`:

1. reward computation can call `state.update()` before auto-reset;
2. terminated envs are reset without increasing `common_step_counter`;
3. observation computation calls `state.update()` again in the same step;
4. because `_last_step == common_step_counter`, PACE state returned early and
   reused stale pre-reset `ball_future_pose`, which could contain NaN;
5. actor `rel_target_base_xy = state.target_base_xy - robot_pos` therefore
   exposed NaN to PPO even though the post-reset physics state was already
   finite.

This is why the run did not look like policy std explosion:
`Mean action std` stayed around `0.60` before failure.

### 修复

- `PingpongPacePredictionState.update()` now recomputes when any
  `episode_length_buf == 0`, even if `common_step_counter` has not changed.
- `_predict_incoming_future_pose()` now applies finite fallback to future pose
  and future time before writing observation-visible PACE target state.
- Added regression test
  `test_pingpong_pace_prediction_refreshes_after_auto_reset_same_step()`.

### 验证

- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/mjlab/tasks/pingpong/rl/runner.py src/mjlab/tasks/pingpong/mdp/pace.py tests/test_pingpong_state.py tests/test_pingpong_task.py`
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ty check src/mjlab/tasks/pingpong tests/test_pingpong_task.py tests/test_pingpong_state.py`
  passed.
- `FORCE_CPU=1 UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pingpong_task.py tests/test_pingpong_state.py -q`
  passed: `25 passed`.
- Fixed diagnostic run:
  `logs/rsl_rl/g1_pingpong_pace/pingpong_pace_nan_diag_fixed_12288env_gpu3_2026-07-07_09-29-44`
  ran `12288` envs for `20` iterations on GPU3 without actor-observation
  NaN and without writing any diagnostics JSON.

### 剩余问题

- PACE direct-control policy quality is still poor at this early stage:
  `fault_reason/body_ball` remains high and `robot_ball_contact_count` is high.
  That is a reward/control-quality issue, separate from this actor-observation
  NaN crash.

## 2026-07-07 09:43 - Restart fixed G1 PACE scratch training

### 目标

After fixing the actor-observation NaN, restart the PACE baseline from scratch
instead of resuming any failed checkpoint.

### 启动配置

- tmux session:
  `pingpong_pace_fixed_scratch_12288_gpu3_20260707`
- task:
  `Mjlab-Pingpong-PACE-Unitree-G1`
- GPU:
  host GPU3 via `CUDA_VISIBLE_DEVICES=3` and `--gpu-ids "[0]"`
- environments:
  `12288` (`12*1024`)
- run name:
  `pingpong_pace_fixed_scratch_12288env_gpu3`
- run dir:
  `logs/rsl_rl/g1_pingpong_pace/pingpong_pace_fixed_scratch_12288env_gpu3_2026-07-07_09-43-26`
- resume:
  `False`

Command:

```sh
CUDA_VISIBLE_DEVICES=3 UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig \
WANDB_CACHE_DIR=/tmp/wandb-cache WANDB_CONFIG_DIR=/tmp/wandb-config \
XDG_CACHE_HOME=/tmp/xdg-cache MJLAB_WARP_QUIET=1 \
uv run train Mjlab-Pingpong-PACE-Unitree-G1 \
  --env.scene.num-envs 12288 \
  --gpu-ids "[0]" \
  --agent.resume False \
  --agent.experiment-name g1_pingpong_pace \
  --agent.run-name pingpong_pace_fixed_scratch_12288env_gpu3 \
  --agent.upload-model False
```

### 启动检查

- Iteration 10 reached without actor-observation NaN.
- No `diagnostics/pace_nan_*` JSON was written in the new run dir.
- GPU3 memory was about `13.8 GiB` for the PACE process.
- `Policy/mean_std` stayed around `0.60` to `0.61`.
- Early behavior is still weak, as expected for scratch direct control:
  body-ball faults remain high and legal returns are still zero.

### 同机 GPU 注意

At launch time, GPU1 was also running an unrelated
`Mjlab-Pingpong-Hit-Unitree-G1` job:
`pingpong_hit_tt_decoder_v2_model30000_15360env_gpu1_20260707`.
Do not confuse it with this PACE baseline run.

## 2026-07-07 13:40 - PACE foot contact sensor and force rewards

### 目标

补齐 `Mjlab-Pingpong-PACE-Unitree-G1` 相比 PACE 参考实现缺失的 foot contact
sensor 相关内容，先只影响 PACE direct-control baseline，不修改 latent Cross /
StrikeQuality、decoder checkpoint、球路分布或当前持拍模型。

### 代码改动

- 新增 PACE-only sensor：`pace_foot_contact`。
  - primary：`left_ankle_roll_link`、`right_ankle_roll_link`
  - secondary：`None`，即脚部任意接触都会被记录
  - `reduce="netforce"`、`history_length=4`、`track_air_time=True`
- PACE reward 从 height-only 支撑近似切到 force-based contact：
  - `pace_fly=-2.5`
  - `pace_hit_unstable_support=-10.0`
  - `feet_slide=-1.5`
  - `feet_force=-3.0e-3`
  - `feet_stumble=-2.0`
- 保留旧的 `pace_fly_height` / `pace_hit_unstable_support_height` helper，
  但 PACE env config 不再使用它们，方便旧脚本兼容。
- 普通 `Mjlab-Pingpong-Cross-Unitree-G1` 和
  `Mjlab-Pingpong-Cross-StrikeQuality-Unitree-G1` 不挂
  `pace_foot_contact`，也不包含上述 PACE foot reward。

### 验证结果

```sh
FORCE_CPU=1 UV_CACHE_DIR=/tmp/uv-cache \
uv run pytest tests/test_pingpong_task.py tests/test_pingpong_state.py -q
# 26 passed

UV_CACHE_DIR=/tmp/uv-cache \
uv run ty check src/mjlab/tasks/pingpong \
  tests/test_pingpong_task.py tests/test_pingpong_state.py
# All checks passed
```

### 注意

已经在 tmux 中运行的旧 PACE 训练不会自动使用新 sensor/reward；需要重启
PACE task 后，新的 env cfg 才会生效。

## 2026-07-07 13:44 - Restart PACE with foot contact rewards

### 目标

停止旧的 `pingpong_pace_fixed_scratch_12288_gpu3_20260707`，重新从 scratch
启动 PACE direct-control baseline，让新 `pace_foot_contact` sensor 和 force-based
foot reward 生效。

### 启动配置

- tmux session:
  `pingpong_pace_foot_contact_12288_gpu3_20260707`
- task:
  `Mjlab-Pingpong-PACE-Unitree-G1`
- GPU:
  host GPU3 via `CUDA_VISIBLE_DEVICES=3` and `--gpu-ids "[0]"`
- environments:
  `12288` (`12*1024`)
- run name:
  `pingpong_pace_foot_contact_12288env_gpu3`
- run dir:
  `logs/rsl_rl/g1_pingpong_pace/pingpong_pace_foot_contact_12288env_gpu3_2026-07-07_13-44-03`
- resume:
  `False`

Command:

```sh
CUDA_VISIBLE_DEVICES=3 UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig \
WANDB_CACHE_DIR=/tmp/wandb-cache WANDB_CONFIG_DIR=/tmp/wandb-config \
XDG_CACHE_HOME=/tmp/xdg-cache MJLAB_WARP_QUIET=1 \
uv run train Mjlab-Pingpong-PACE-Unitree-G1 \
  --env.scene.num-envs 12288 \
  --gpu-ids "[0]" \
  --agent.resume False \
  --agent.experiment-name g1_pingpong_pace \
  --agent.run-name pingpong_pace_foot_contact_12288env_gpu3 \
  --agent.upload-model False
```

### 启动检查

- 新 run 已到 iteration 2。
- stdout 已出现新的 `Episode_Reward/pace_fly`、
  `Episode_Reward/pace_hit_unstable_support`、`Episode_Reward/feet_slide`、
  `Episode_Reward/feet_force`、`Episode_Reward/feet_stumble`，说明新 reward
  生效。
- `Episode_Termination/nan_detection=0.0000`。
- `Policy/mean_std` 仍在 `0.60` 左右。
