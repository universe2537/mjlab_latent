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
