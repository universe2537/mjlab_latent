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
