.. _motion-imitation:

Motion Imitation
================

mjlab can train humanoid policies to imitate reference motions. This page
covers motion data preprocessing and training.

WandB registry setup
--------------------

mjlab uses `Weights & Biases <https://wandb.ai/>`_ to store and load
reference motions. Before preprocessing any motions, create a WandB registry
by following the
`BeyondMimic instructions <https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup>`_
(only the registry creation step; skip the ``csv_to_npz.py`` command shown
there).

Motion preprocessing
--------------------

Reference motions are retargeted CSV files in Unitree's generalized
coordinate convention (base position, base quaternion in xyzw, then joint
angles).

Convert a CSV to the NPZ format mjlab expects:

.. code-block:: bash

   MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
       --input-file <PATH_TO_CSV> \
       --output-name <MOTION_NAME> \
       --input-fps 30 \
       --output-fps 50 \
       --render True

The script plays the motion through MuJoCo Warp, computes forward kinematics
for every body, and uploads the resulting NPZ to your WandB registry.

.. warning::

   You **must** use mjlab's converter (``mjlab.scripts.csv_to_npz``).
   Converters from other frameworks such as IsaacLab produce NPZ files with
   incompatible body orderings. The NPZ stores precomputed body positions and
   quaternions indexed by body number, and different physics engines assign
   body indices differently (MuJoCo uses depth first traversal, PhysX uses
   breadth first). A mismatched NPZ will map tracking targets to the wrong
   bodies and training will not converge.

Training
--------

Tracking tasks read the motion source from the task configuration. For the
Unitree G1 tracking task this is configured in
``src/mjlab/tasks/tracking/config/g1/env_cfgs.py``:

.. code-block:: python

   motion_cmd.motion_source = "local"
   motion_cmd.motion_files = "data/LAFAN/g1/npz/dance1_subject1.npz"

``motion_files`` accepts either a single string or a tuple of strings, and is
interpreted according to ``motion_source``. Use ``motion_source="local"`` to load
``motion_files`` as local NPZ paths. The paths may use environment variables
such as ``${GLI_PATH}``. Use ``motion_source="wandb"`` to load ``motion_files``
as W&B artifact paths:

.. code-block:: python

   motion_cmd.motion_source = "wandb"
   motion_cmd.motion_files = (
       "motions/dance1_subject1",
       "motions/dance2_subject1",
   )

Values like ``WANDB_PROJECT``, ``WANDB_ENTITY``, and ``GLI_PATH`` can be placed
in a local ``.env`` file. When a W&B ``motion_files`` entry omits the entity,
training uses ``WANDB_ENTITY`` from ``.env``.

.. code-block:: bash

   uv run train Mjlab-Tracking-Flat-Unitree-G1 \
       --env.scene.num-envs 4096

Evaluation
----------

.. code-block:: bash

   uv run play Mjlab-Tracking-Flat-Unitree-G1 \
       --wandb-run-path your-org/mjlab/run-id
