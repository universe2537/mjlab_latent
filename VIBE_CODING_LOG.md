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
