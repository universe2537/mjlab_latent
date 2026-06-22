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
