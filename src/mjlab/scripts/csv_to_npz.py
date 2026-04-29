"""把 retargeted CSV motion 转成 mjlab tracking 任务使用的 NPZ。

输入:
  CSV 文件。每一行是一帧, 格式为:
  base_pos(3), base_quat_xyzw(4), joint_pos(N)。

输出:
  1. 本地临时文件 /tmp/motion.npz。
  2. W&B registry 中的 motions/<output_name> artifact。

核心流程:
  读取 CSV -> 插值到 output_fps -> 计算速度 -> 写入 MuJoCo 状态
  -> forward kinematics -> 保存 joint/body 的 pose 和 velocity。
"""

from typing import Any

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


class MotionLoader:
  """加载、重采样并逐帧提供 motion 数据。

  这个类不直接和 MuJoCo 交互。它只负责把 CSV 中的 root/joint 轨迹
  整理成 run_sim 可以逐帧写入仿真的 tensor。
  """

  def __init__(
    self,
    motion_file: str,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    line_range: tuple[int, int] | None = None,
  ):
    """初始化 motion loader。

    输入:
      motion_file: CSV 文件路径。
      input_fps: CSV 原始帧率。
      output_fps: 输出给 mjlab 的目标帧率。
      device: tensor 所在设备, 例如 "cuda:0" 或 "cpu"。
      line_range: 可选的 1-based 行范围, 只读取 CSV 的一部分。

    输出:
      不 return。初始化时会生成插值后的位姿、关节角和速度缓存。
    """
    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    # dt 是相邻两帧的时间间隔, 后面插值和求速度都依赖它。
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = device
    self.line_range = line_range
    # 三步预处理: 读 CSV、重采样、根据轨迹求速度。
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    """从 CSV 读取原始 motion。

    输入:
      self.motion_file 指向的 CSV。

    输出:
      self.motion_base_poss_input: shape (T, 3), root/base 位置。
      self.motion_base_rots_input: shape (T, 4), root/base 姿态, wxyz。
      self.motion_dof_poss_input: shape (T, num_joints), 关节角。
      self.input_frames: 输入帧数。
      self.duration: motion 持续时间。
    """
    if self.line_range is None:
      # 不指定 line_range 时读取整个 CSV。
      motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
    else:
      # line_range 是 1-based 闭区间, 所以 skiprows 要减 1。
      motion = torch.from_numpy(
        np.loadtxt(
          self.motion_file,
          delimiter=",",
          skiprows=self.line_range[0] - 1,
          max_rows=self.line_range[1] - self.line_range[0] + 1,
        )
      )
    # 后续计算使用 torch tensor, 并移动到指定 CPU/GPU。
    motion = motion.to(torch.float32).to(self.device)
    # motion[:, 2] -= 0.05
    # CSV 前 3 列是 root/base 的世界系位置。
    self.motion_base_poss_input = motion[:, :3]
    # CSV 第 4-7 列是 quaternion, 原始约定是 xyzw。
    self.motion_base_rots_input = motion[:, 3:7]
    self.motion_base_rots_input = self.motion_base_rots_input[
      :, [3, 0, 1, 2]
    ]  # 转成 mjlab/MuJoCo 使用的 wxyz。
    # 第 8 列之后是关节位置, 顺序必须和 run_sim 里的 joint_names 一致。
    self.motion_dof_poss_input = motion[:, 7:]

    self.input_frames = motion.shape[0]
    # T 帧之间只有 T - 1 个时间间隔, 所以持续时间不是 T * dt。
    self.duration = (self.input_frames - 1) * self.input_dt

  def _interpolate_motion(self):
    """把 motion 从 input_fps 重采样到 output_fps。

    输入:
      _load_motion 生成的原始 root/joint 轨迹。

    输出:
      self.motion_base_poss: 插值后的 root/base 位置。
      self.motion_base_rots: 插值后的 root/base quaternion。
      self.motion_dof_poss: 插值后的关节位置。
      self.output_frames: 输出帧数。
    """
    # 生成输出帧的时间戳, 例如 50 FPS 对应 0.00, 0.02, 0.04, ...
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    # 对每个输出时间点, 找到它位于哪两个输入帧之间以及插值比例。
    index_0, index_1, blend = self._compute_frame_blend(times)
    # 位置是欧氏空间向量, 可以直接线性插值。
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    # 姿态是单位 quaternion, 需要球面插值, 不能普通线性插值。
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    # 关节角在这里按标量序列处理, 使用线性插值。
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )
    print(
      f"Motion interpolated, input frames: {self.input_frames}, "
      f"input fps: {self.input_fps}, "
      f"output frames: {self.output_frames}, "
      f"output fps: {self.output_fps}"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """线性插值。

    输入:
      a: 左端点。
      b: 右端点。
      blend: 插值比例, 通常在 [0, 1]。

    输出:
      a * (1 - blend) + b * blend。
    """
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """对 quaternion 做球面线性插值。

    输入:
      a: shape (T, 4), 左端 quaternion, wxyz。
      b: shape (T, 4), 右端 quaternion, wxyz。
      blend: shape (T,), 每一帧的插值比例。

    输出:
      shape (T, 4), 插值后的 quaternion。
    """
    slerped_quats = torch.zeros_like(a)
    for i in range(a.shape[0]):
      # quat_slerp 一次处理一对 quaternion, 所以这里逐帧循环。
      slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return slerped_quats

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算每个输出时间点对应的输入帧和插值比例。

    输入:
      times: shape (T,), output_fps 下的时间戳。

    输出:
      index_0: shape (T,), 左侧输入帧 index。
      index_1: shape (T,), 右侧输入帧 index。
      blend: shape (T,), index_0 到 index_1 的插值比例。
    """
    # phase 把绝对时间归一化到 [0, 1]。
    phase = times / self.duration
    # 把 phase 映射回输入帧坐标, floor 后得到左侧帧。
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    # 右侧帧不能超过最后一帧。
    index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
    # 小数部分就是插值比例, 例如 12.7 -> frame 12 和 13 之间的 0.7。
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _compute_velocities(self):
    """根据插值后的轨迹计算速度。

    输入:
      self.motion_base_poss, self.motion_dof_poss, self.motion_base_rots。

    输出:
      self.motion_base_lin_vels: root/base 线速度。
      self.motion_dof_vels: 关节速度。
      self.motion_base_ang_vels: root/base 角速度。
    """
    # 对位置沿时间维度求导, 得到线速度。
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    # 对关节角沿时间维度求导, 得到关节速度。
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    # quaternion 不能直接普通求导, 需要在 SO(3) 上计算角速度。
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """在 SO(3) 上根据 quaternion 序列计算角速度。

    输入:
      rotations: shape (B, 4), 每一帧的 root/base quaternion, wxyz。
      dt: 相邻两帧的时间间隔。

    输出:
      shape (B, 3), 每一帧的角速度。
    """
    # 中心差分: 用 i-1 和 i+1 两帧估计第 i 帧角速度。
    q_prev, q_next = rotations[:-2], rotations[2:]
    # 相对旋转 q_rel = q_next * inverse(q_prev)。
    # 单位 quaternion 的 inverse 等于 conjugate。
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B-2, 4)

    # q_prev 到 q_next 跨过两个 dt, 所以角速度要除以 2 * dt。
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B-2, 3)
    omega = torch.cat(
      [omega[:1], omega, omega[-1:]], dim=0
    )  # 首尾无法中心差分, 用相邻值补齐。
    return omega

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    """返回当前帧 motion state, 并把内部 index 前进一帧。

    输入:
      无显式参数, 使用 self.current_idx。

    输出:
      state: 当前帧的 root pose/velocity 和 joint pose/velocity。
      reset_flag: 如果已经读完整段 motion, 则为 True。
    """
    # 用切片而不是直接索引, 是为了保留 batch 维度: (1, ...).
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    # 读到最后后重置, 让外层 run_sim 知道可以保存并结束。
    if self.current_idx >= self.output_frames:
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


def run_sim(
  sim: Simulation,
  scene: Scene,
  joint_names,
  input_file,
  input_fps,
  output_fps,
  output_name,
  render,
  line_range,
  renderer: OffscreenRenderer | None = None,
):
  """把 motion 写入 MuJoCo, 生成 mjlab tracking 需要的 NPZ。

  输入:
    sim: 已创建的 mjlab Simulation。
    scene: 已初始化的 mjlab Scene。
    joint_names: CSV 中关节角对应的 G1 关节名顺序。
    input_file: CSV 文件路径。
    input_fps: CSV 原始帧率。
    output_fps: 输出 NPZ 的帧率。
    output_name: 上传到 W&B motions registry 的 artifact 名。
    render: 是否额外渲染 motion.mp4。
    line_range: 可选的 CSV 行范围。
    renderer: render=True 时使用的离屏渲染器。

  输出:
    不 return。副作用是保存 /tmp/motion.npz, 并上传到 W&B registry。
  """
  # MotionLoader 负责读 CSV、插值和求速度。
  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    line_range=line_range,
  )

  # 从 scene 中取出机器人实体。
  robot: Entity = scene["robot"]
  # 找到 motion 关节在机器人完整关节数组中的 index。
  # preserve_order=True 保证 CSV 关节顺序和 joint_names 顺序一一对应。
  robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

  # 这些 key 就是最终 motion.npz 中的字段。
  log: dict[str, Any] = {
    "fps": [output_fps],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  file_saved = False

  # render=True 时, 每一帧图像会暂存在 frames, 最后合成视频。
  frames = []
  scene.reset()

  print(f"\nStarting simulation with {motion.output_frames} frames...")
  if render:
    print("Rendering enabled - generating video frames...")

  # 创建进度条, 用来显示逐帧 replay 的进度。
  pbar = tqdm(
    total=motion.output_frames,
    desc="Processing frames",
    unit="frame",
    ncols=100,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  )

  frame_count = 0
  while not file_saved:
    # 每次从 MotionLoader 取一帧 root/joint 状态。
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    # root_states 包含 root/base 的位置、姿态、线速度和角速度。
    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    # 多环境仿真时每个环境可能有不同 origin; 这里虽然只有 1 个环境,
    # 仍然沿用 mjlab 的 batch/env 数据布局。
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = motion_base_ang_vel
    # 把 root/base 状态写入 MuJoCo data。
    robot.write_root_state_to_sim(root_states)

    # 从默认关节状态开始, 只覆盖 motion CSV 提供的那些关节。
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = motion_dof_pos
    joint_vel[:, robot_joint_indexes] = motion_dof_vel
    # 把关节位置和速度写入 MuJoCo data。
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    # forward 不做积分推进, 只根据当前状态计算 forward kinematics。
    sim.forward()
    # 把 MuJoCo 计算出的 body/link 数据同步到 mjlab scene cache。
    scene.update(sim.mj_model.opt.timestep)
    if render and renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    if not file_saved:
      # 记录完整机器人关节状态, 而不仅是 CSV 中出现的关节。
      log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
      # 记录每个 body/link 在世界坐标系下的位置和姿态。
      log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
      log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
      # 记录每个 body/link 在世界坐标系下的线速度和角速度。
      log["body_lin_vel_w"].append(
        robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
      )
      log["body_ang_vel_w"].append(
        robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
      )

      # sanity check: 第 0 个 body 通常对应 root/base, 它的速度应与输入一致。
      torch.testing.assert_close(
        robot.data.body_link_lin_vel_w[0, 0],
        motion_base_lin_vel[0],
        atol=1e-4,
        rtol=1e-4,
      )
      torch.testing.assert_close(
        robot.data.body_link_ang_vel_w[0, 0],
        motion_base_ang_vel[0],
        atol=1e-4,
        rtol=1e-4,
      )

      frame_count += 1
      pbar.update(1)

      if frame_count % 100 == 0:  # 每 100 帧更新一次描述, 避免刷屏。
        elapsed_time = frame_count / output_fps
        pbar.set_description(f"Processing frames (t={elapsed_time:.1f}s)")

      if reset_flag and not file_saved:
        # reset_flag=True 表示 MotionLoader 已经读完一整段 motion。
        file_saved = True
        pbar.close()

        print("\nStacking arrays and saving data...")
        # list[frame_array] -> ndarray, 第一维是时间 T。
        for k in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        ):
          log[k] = np.stack(log[k], axis=0)

        # 本地文件名是固定的, 多次运行会覆盖 /tmp/motion.npz。
        print("Saving to /tmp/motion.npz...")
        np.savez("/tmp/motion.npz", **log)

        print("Uploading to Weights & Biases...")
        import wandb

        # output_name 是 W&B artifact/registry 中的 motion 名称。
        COLLECTION = output_name
        run = wandb.init(project="csv_to_npz", name=COLLECTION)
        print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
        REGISTRY = "motions"
        # 上传本地 NPZ 为 artifact, 类型固定为 motions。
        logged_artifact = run.log_artifact(
          artifact_or_path="/tmp/motion.npz", name=COLLECTION, type=REGISTRY
        )
        # 链接到 W&B Registry, 训练时通过 your-org/motions/<name> 读取。
        run.link_artifact(
          artifact=logged_artifact,
          target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}",
        )
        print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION}")

        if render:
          import mediapy as media

          # 本地视频名也是固定的, 多次 render 会覆盖 ./motion.mp4。
          print("Creating video...")
          media.write_video("./motion.mp4", frames, fps=output_fps)

          # 视频只作为可视化记录上传, 训练真正使用的是 motion.npz。
          print("Logging video to wandb...")
          wandb.log({"motion_video": wandb.Video("./motion.mp4", format="mp4")})

        wandb.finish()


def main(
  input_file: str,
  output_name: str,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  line_range: tuple[int, int] | None = None,
):
  """命令行入口: replay CSV motion 并输出 mjlab 需要的 NPZ。

  输入:
    input_file: CSV 输入文件路径。
    output_name: W&B motions registry 中使用的 motion 名。
    input_fps: CSV 原始帧率。
    output_fps: 输出 NPZ 的目标帧率。
    device: 计算设备, 默认 "cuda:0"。
    render: 是否渲染并上传 motion.mp4。
    line_range: 可选的 CSV 行范围, 用于只处理片段。

  输出:
    不 return。最终生成 /tmp/motion.npz 并上传到 W&B motions registry。
  """
  # 如果请求 CUDA 但当前机器没有 CUDA, 自动回退到 CPU。
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA is not available. Falling back to CPU. This may be slow.")
    device = "cpu"

  # MuJoCo timestep 和 output_fps 对齐, 这样每一帧 replay 正好是一个输出帧。
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  # 创建 Unitree G1 的 flat tracking scene。
  scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
  # compile 会把 mjlab scene 转成 MuJoCo 可执行的 model。
  model = scene.compile()

  # 只需要单环境 replay motion, 所以 num_envs=1。
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

  # 初始化 scene, 让 scene 持有 MuJoCo model/data 的引用。
  scene.initialize(sim.mj_model, sim.model, sim.data)

  renderer = None
  if render:
    # 离屏渲染配置。视频只用于检查 motion, 不参与训练。
    viewer_cfg = ViewerConfig(
      height=480,
      width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="robot",
      distance=2.0,
      elevation=-5.0,
      azimuth=20,
    )
    # 创建离屏渲染器, 适合 MUJOCO_GL=egl 的无显示器环境。
    renderer = OffscreenRenderer(
      model=sim.mj_model,
      cfg=viewer_cfg,
      scene=scene,
    )
    renderer.initialize()

  # joint_names 定义 CSV 第 8 列之后的关节角如何映射到 G1 机器人。
  # 如果 CSV 的关节顺序和这里不同, 生成的 motion.npz 会错位。
  run_sim(
    sim=sim,
    scene=scene,
    joint_names=[
      "left_hip_pitch_joint",
      "left_hip_roll_joint",
      "left_hip_yaw_joint",
      "left_knee_joint",
      "left_ankle_pitch_joint",
      "left_ankle_roll_joint",
      "right_hip_pitch_joint",
      "right_hip_roll_joint",
      "right_hip_yaw_joint",
      "right_knee_joint",
      "right_ankle_pitch_joint",
      "right_ankle_roll_joint",
      "waist_yaw_joint",
      "waist_roll_joint",
      "waist_pitch_joint",
      "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint",
      "left_shoulder_yaw_joint",
      "left_elbow_joint",
      "left_wrist_roll_joint",
      "left_wrist_pitch_joint",
      "left_wrist_yaw_joint",
      "right_shoulder_pitch_joint",
      "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint",
      "right_elbow_joint",
      "right_wrist_roll_joint",
      "right_wrist_pitch_joint",
      "right_wrist_yaw_joint",
    ],
    input_fps=input_fps,
    input_file=input_file,
    output_fps=output_fps,
    output_name=output_name,
    render=render,
    line_range=line_range,
    renderer=renderer,
  )


if __name__ == "__main__":
  # tyro 会把 main(...) 的参数自动暴露成 CLI 参数。
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
