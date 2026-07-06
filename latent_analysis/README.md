# Table-Tennis Latent Analysis

This folder contains an offline diagnostic for the table-tennis distillation
checkpoint. It does not change training code, checkpoints, or the Pingpong
default decoder.

Run the table-tennis checkpoint:

```sh
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig \
uv run python latent_analysis/analyze_latent_space.py \
  --checkpoint /data0/universe/home_moved/mjlab_latent/logs/rsl_rl/g1_distillation_table_tennis/table_tennis_distill_v1_46080env_from_tracking18000_2026-07-03_10-14-26/model_30000.pt \
  --motion-root artifacts/table_tennis \
  --output-dir latent_analysis/outputs \
  --samples-per-motion 1024
```

Run the current default tennis decoder checkpoint:

```sh
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig \
uv run python latent_analysis/analyze_latent_space.py \
  --analysis-label tennis-default \
  --task-id Mjlab-Distill-Flat-Unitree-G1 \
  --checkpoint /data0/universe/home_moved/mjlab_latent/logs/rsl_rl/g1_distillation/distill_cloud_unitree_racket_tennis_2026-05-12_09-35-14/model_30000.pt \
  --motion-root artifacts \
  --motion-pattern 'tennis_random_*/motion.npz' \
  --output-dir latent_analysis/outputs_tennis \
  --samples-per-motion 1024
```

The script defaults to CPU and uses a teacher-forced reference-state probe:
it loads the selected distillation environment, balances samples across matching
`motion.npz` files, queries the frozen tracking
teacher for action labels, and then evaluates the student posterior, prior, and
decoder on the same observations.

Outputs:

- `metrics_summary.json` / `metrics_summary.md`: final, best, and last100
  TensorBoard scalars for distillation losses and rollout settings.
- `latent_stats.json`: active dimensions, per-dimension ranges, KL, mean L2
  prior/posterior alignment, and covariance spectra.
- `reconstruction_metrics.json`: posterior/prior action MSE and MAE globally,
  by split, by motion, and by joint.
- `per_motion_metrics.csv`: compact per-motion table for train, held-out
  `test_001`, and diagnostic `zhengshou_002_badend`.
- `report.md` / `report.html`: final verdict and figures.
- `*.png`: PCA, activity, alignment, and reconstruction plots.

How to read the results:

- `prior` is the deployable path. If prior reconstruction is close to
  posterior reconstruction and held-out `test_001` is close to train motions,
  the compressed latent controller is behaving like the teacher on reference
  states.
- `posterior` sees tracking targets and is the easier training-time path. A
  large posterior/prior gap means the prior may not yet reproduce the encoder.
- Active latent dimensions are counted from posterior/prior mean variation
  across samples. If most dimensions are nearly constant, the latent bottleneck
  is underused.
- PCA plots are structure checks, not proof. Useful structure is continuous by
  phase or clustered by action family; weak structure should be treated as a
  caution sign, then verified with rollout videos.
- The report verdict is one of `usable`, `usable_with_caution`, or `not_ready`.
  Even `usable` still needs closed-loop Pingpong play validation before
  changing `DEFAULT_DECODER_CHECKPOINT`.

Optional plots:

```sh
uv run python latent_analysis/analyze_latent_space.py --tsne
uv run python latent_analysis/analyze_latent_space.py --umap
```

Those flags only run if `sklearn` or `umap-learn` are already installed.
