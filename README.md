# Language-Conditioned VLA Robot Manipulation

A vision-language-action (VLA) system that drives an **SO-101** robot arm from a
camera image + a natural-language instruction ("*put the red block in the green
bowl*") + the arm's joint state, and outputs the continuous joint commands to do
it. It spans the full loop: **teleop data collection → dataset → policy training
(BC → ACT → SmolVLA → OpenVLA) → async inference server + safety filter → an
evaluation suite with confidence intervals → ablations + failure analysis.**

> **New here? Read [`GUIDE.md`](GUIDE.md)** — it explains the project from the
> ground up (Section 1 assumes only first-year CS), what to learn to build it
> yourself (Section 2), a file-by-file interview-grade walkthrough (Section 3),
> the portfolio write-up (Section 4), and résumé bullets (Section 5).

<!-- IMAGE: hero shot — the arm mid-grasp over the gridded mat, front+wrist camera
     insets, instruction text overlaid. Take a still from demo_successes.mp4. -->

---

## Architecture

```
 front + wrist RGB ─┐
 instruction text ──┼─▶  VLA backbone / encoders  ─▶  policy head  ─▶  action chunk
 joint state ───────┘        (on the GPU box)                              │
                                                                           ▼
                                              temporal ensembling ─▶ SAFETY FILTER ─▶ motors
                                                               (joint+velocity clamp,
                                                                FK workspace box, e-stop)
```

The model runs in a **policy server** on a GPU machine; a thin **client** on the
robot machine streams observations over a WebSocket and executes the returned
action chunk at 30 Hz, requesting the next chunk before the current one runs out
so GPU latency is hidden.

<!-- DIAGRAM: redraw the above as a clean figure (draw.io / Excalidraw), export
     PNG to docs/architecture.png, and embed it here. -->

---

## Repository layout

```
configs/            YAML/JSON: robot geometry, safety envelope, training recipes,
                    paraphrase bank, example eval suite
src/vla/
  data/             LeRobotDataset adapter, BC torch Dataset, normalization stats,
                    seeded layout randomization, paraphrase bank
  models/           BC baseline: frozen CLIP text + ResNet-18 image + MLP head
  training/         BC training loop
  inference/        policy server, robot client, temporal ensembler, SAFETY FILTER,
                    per-policy wrappers (bc/act/smolvla/openvla), msgpack wire codec
  eval/             trial-spec schema, Wilson-CI metrics, operator-in-loop harness
  robot/            SO-101 forward kinematics, LeRobot robot iface, ROS2 bridge,
                    hardware-free DummyRobot
  utils/            seeding, YAML config + overrides, experiment logging
scripts/            CLI entrypoints (train, collect_demos, teleop, evaluate,
                    run_policy_server, convert_to_openvla, demo_offline, ...)
tests/              47 unit/integration tests for the hardware-free logic
notebooks/          dataset_visualization.ipynb (pre-training sanity checks)
reports/            final_report.md (ablation + failure tables), model_card.md
```

---

## Quickstart on ANY machine (no GPU, no robot)

The control-loop logic, safety filter, metrics, and evaluation harness all run on
CPU. This is the first thing to try after cloning.

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install numpy scipy pyyaml pytest                 # minimal deps for the offline path

python -m pytest -q                                   # 47 tests, all hardware-free
python scripts/demo_offline.py --ticks 60             # full loop w/ DummyRobot + safety filter
python scripts/evaluate.py --suite configs/eval_trials_example.json --dry-run
python scripts/generate_randomization_sheet.py \
    --num-episodes 50 --objects "red block" "green bowl" --out data/layouts/task1.csv
```

`demo_offline.py` deliberately injects out-of-range targets so you can watch the
safety filter clamp them — no unsafe command reaches the (simulated) motors.

---

## Full setup on a NEW GPU machine (training + real robot)

This is the box that actually trains models and/or runs the arm. Steps assume
Ubuntu 22.04 + an NVIDIA GPU (≥24 GB for the VLAs), or WSL2 on Windows.

### 1. System prerequisites
```bash
# NVIDIA driver + CUDA toolkit must already be installed; verify:
nvidia-smi                       # should list your GPU
python --version                 # need 3.10+
```

### 2. Environment
```bash
git clone <your-fork-url> vla-robot-manipulation-project
cd vla-robot-manipulation-project

conda create -n vla python=3.10 -y && conda activate vla
# Install a CUDA build of torch FIRST (match your CUDA version):
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .                 # makes `vla` importable + installs console scripts
```
> `bitsandbytes` (QLoRA) is Linux/CUDA-only; on Windows use WSL2 or skip QLoRA.

### 3. Accounts / tracking
```bash
huggingface-cli login            # datasets + checkpoints push here
wandb login                      # experiment tracking (or set logging.backend: tensorboard)
```

### 4. (Real robot only) find ports, calibrate, mount cameras
```bash
lerobot-find-port                                        # identify each arm's serial port
lerobot-calibrate --robot.type=so101_follower --robot.port=<PORT>
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=<PORT>
# Copy the calibrated joint limits into configs/robot_so101.yaml.
# Set the two camera indices in configs/robot_so101.yaml, then sanity-check:
python scripts/teleop.py --robot-port <PORT> --teleop-port <PORT>
```

### 5. Sanity-check the install (no robot needed)
```bash
python -m pytest -q
python scripts/demo_offline.py
```

---

## The pipeline (end to end)

```bash
# 1. Generate a reproducible object-placement sheet, then collect teleop demos.
python scripts/generate_randomization_sheet.py --num-episodes 50 \
    --objects "red block" "green bowl" --out data/layouts/block_in_bowl.csv
python scripts/collect_demos.py --task block_in_bowl \
    --repo-id youruser/so101_tabletop_v1 --num-episodes 50 \
    --robot-port <PORT> --teleop-port <PORT> --layout-sheet data/layouts/block_in_bowl.csv

# 2. Inspect the dataset before training.
jupyter lab notebooks/dataset_visualization.ipynb

# 3. Train. Same entrypoint, different --policy.
python scripts/train.py --policy bc      --config configs/train_bc.yaml
python scripts/train.py --policy act     --config configs/train_act.yaml
python scripts/train.py --policy smolvla --config configs/train_smolvla.yaml
# OpenVLA: convert first, then fine-tune (needs the OpenVLA-OFT repo installed).
python scripts/convert_to_openvla.py --repo-id youruser/so101_tabletop_v1 \
    --out data/openvla_rlds/so101_tabletop_v1
python scripts/train.py --policy openvla --config configs/train_openvla_lora.yaml

# 4. Serve the trained policy on the GPU box.
python scripts/run_policy_server.py --policy-kind bc \
    --checkpoint outputs/bc_baseline_v1/bc_best.pt --port 8000

# 5. Evaluate on the robot (client connects to the server). Same suite for every model.
python scripts/evaluate.py --suite configs/eval_trials_example.json \
    --server-uri ws://<gpu-box>:8000
```

The evaluation harness prints per-condition success with **Wilson 95% confidence
intervals**, the **paraphrase gap**, and writes `trials.csv` + `report.md`.

---

## Reproducibility

- All seeds fixed at **42**; `set_seed` covers python/numpy/torch + cuDNN.
- Every experiment's settings live in a committed YAML; CLI `--override key=val`
  layers on top without editing the file.
- Dataset versions are **tagged** on the Hub (`v1`, `v2`, …) and never overwritten.
- **Record the exact LeRobot commit hash** in this README — the API moves fast:
  `LeRobot commit: [paste hash here]`.

## Safety

The `SafetyFilter` (`src/vla/inference/safety.py`) sits between the policy and the
motors and is **non-negotiable on real hardware**: per-joint limit clamps,
per-tick velocity clamps, an FK-based workspace bounding box (rejects → holds the
last safe pose), a gripper-current cap, and a keyboard e-stop that torques off all
servos. See `reports/model_card.md` for its scope and limits.

## License
MIT (this code). Base VLA weights (OpenVLA, SmolVLA) are under their own licenses.
