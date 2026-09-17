# Language-Conditioned VLA Manipulation — Final Report

> **Evaluation scope (read first).** Every model here was trained and evaluated
> **entirely offline, on a public dataset, with no robot.** The main metric measured is
> the held-out **action-prediction error**, not task success as the policy was never executed on real hardware.

## 1. Problem & setup

We train a policy mapping **(front + wrist RGB, natural-language instruction,
6-DoF joint state) → next joint-target action** for an SO-101 arm doing tabletop
pick-and-place, and compare three model families on identical data and splits.

- **Data:** [`youliangtan/so101-table-cleanup`](https://huggingface.co/datasets/youliangtan/so101-table-cleanup)
  — a public, third-party SO-101 dataset (not collected by the author).
  **80 episodes, 46,963 frames, 4 language tasks**, front + wrist cameras at
  640×480, 30 fps, `so101_follower`.
- **Tasks:** "Grab {pens, markers, tapes, scissor} and place into pen holder."
  All four share one scene, so **the instruction is the only signal that
  distinguishes them** — a clean setting for language conditioning.
- **Split:** 72 train / 8 held-out episodes, split **by episode** (never by
  frame, which would leak temporally-adjacent frames across the boundary),
  `seed=42`. Held out: episodes `[0, 25, 28, 33, 40, 51, 60, 61]`. All three
  models use this same split (`src/vla/data/splits.py`) — LeRobot's trainer
  defaults to using *every* episode, so the training entrypoint passes it an
  explicit episode list. Without that, ACT and SmolVLA would have trained on the
  episodes they are scored on.
- **Compute:** RTX 5070 (12 GB, Blackwell), driver 581.42, torch 2.7.0+cu128,
  `lerobot==0.4.1`, seeds fixed at 42. Total training 14 h 14 min.

### Data integrity note
The published dataset's `meta/info.json` declares `total_frames: 47513`, but the
actual data — in both the v2.1 original and the v3.0 conversion — contains
**46,963** frames across 80 episodes. LeRobot's sampler trusts the metadata and
indexes past the end of the table (`IndexError: Invalid key: 47431 is out of
bounds for size 46963`). The count was corrected locally before training. Our own
BC loader was unaffected because it derives frame counts from the table itself.

## 2. Models compared

| Model | Trainable params | Action head | Chunk | Budget | Notes |
|-------|------------------|-------------|-------|--------|-------|
| BC baseline (ours) | 12.4M | MLP, single step | 1 | 20 epochs / 256 min | frozen CLIP text + ResNet-18 + MLP |
| ACT | 51.6M | CVAE | 100 | 100k steps / 297 min | trained from scratch |
| SmolVLA | 450M | flow matching | 50 | 20k steps / 299 min | fine-tuned from `lerobot/smolvla_base` |
| OpenVLA-OFT | 7B | — | — | **not trained** | does not fit in 12 GB; see §7 |

**Training losses are not comparable across these rows.** BC minimises MSE on
normalised actions (final held-out MSE 0.0632), ACT a CVAE objective including a
KL term (final 0.074), SmolVLA a flow-matching loss (final 0.010). Different
objectives on different scales — SmolVLA's loss being numerically smallest means
nothing on its own. §3 exists precisely to put all three on one metric.

![training curves](../docs/training_curves.png)

## 3. Headline results — held-out action-prediction error

Mean absolute error between the policy's action and the demonstrator's, on
**989 frames from the 8 held-out episodes** (every 5th frame), open-loop, single
step, in physical units (degrees). Identical frames, identical order, for every
model. CIs are 2,000-sample bootstrap over frames.

| Model | MAE ↓ | 95% CI | Median AE | RMSE | vs. BC |
|-------|-------|--------|-----------|------|--------|
| BC (from scratch) | 5.128 | [4.986, 5.276] | 4.808 | 7.244 | — |
| ACT | 2.151 | [2.090, 2.214] | 2.005 | 3.069 | **−58.1%** |
| **SmolVLA (fine-tuned)** | **1.736** | [1.673, 1.797] | 1.539 | 2.697 | **−66.1%** |

The three confidence intervals do not overlap, so the ordering
BC > ACT > SmolVLA is statistically solid on this data.

![results](../docs/results.png)

### Per-task breakdown

| Model | markers | pens | scissor | tapes |
|-------|---------|------|---------|-------|
| BC | 4.530 | 5.197 | 6.246 | 5.082 |
| ACT | 2.212 | 1.676 | 2.550 | 2.054 |
| SmolVLA | 1.797 | 1.375 | 1.930 | 1.732 |

The ranking is identical on all four tasks, so the aggregate is not being driven
by one easy task. "Scissor" is hardest for every model and "pens" easiest for the
two chunked models — consistent with scissors being the most geometrically
awkward grasp in this scene, though we cannot confirm that without rollouts.

### Limitations of this result
Held-out action error is measured **open-loop**: at every scored frame the policy
is shown a *demonstrator's* state, never a state produced by its own earlier
actions. Additionally, a policy could also track the average trajectory closely while never
actually closing the gripper. As a result, on a real robot, a model which predicted the human's movement very closely may not necessarily complete the task successfully.

## 4. Inference latency and the cost of chunking

Measured on an idle RTX 5070, 50 timed calls after warm-up, chunk queue reset
before each call so every measurement is a true forward pass. A deployed loop
runs at **30 Hz** (the dataset's capture rate), so one query must complete before
its chunk finishes executing.

| Model | mean | p95 | chunk | motion supplied | duty cycle | fits 30 Hz? |
|-------|------|-----|-------|-----------------|-----------|-------------|
| BC | 14.2 ms | 14.9 ms | 1 | 33 ms | 42.7% | yes |
| BC *without text cache* | 33.8 ms | 35.3 ms | 1 | 33 ms | **101.4%** | **no** |
| ACT | 45.2 ms | 47.0 ms | 100 | 3,333 ms | **1.4%** | yes |
| SmolVLA | 802.0 ms | 864.0 ms | 50 | 1,667 ms | 48.1% | yes |

Notes:

1. **Chunking does not raise the control rate — it buys time.** The arm runs at
   30 Hz regardless. ACT's 45 ms query supplies 3.3 s of motion, so it spends
   **1.4%** of wall-clock computing; it could run on far weaker hardware. (Naively
   reporting "chunk ÷ latency" as an effective rate would yield ACT ≈ 2,200 Hz,
   a figure no robot could realise. It is not reported.)
2. **SmolVLA is 56× slower per query than BC** yet still fits its budget, purely
   because chunking gives it 1.67 s to produce an answer. At 48.1% duty it has
   the least headroom of the three — a slower GPU, or a smaller `n_action_steps`,
   would break it.
3. **A one-line optimisation decided whether the baseline was deployable at
   all.** The BC policy re-ran the frozen CLIP text tower on every control tick,
   even though the instruction is constant for an entire episode. Caching the
   embedding by string made it **2.38× faster (33.8 → 14.2 ms)** — and took the
   duty cycle from 101.4% (cannot sustain 30 Hz) to 42.7% (comfortable).

### Loop and safety-filter timing (no robot)
`scripts/demo_offline.py` runs the full control loop against a simulated robot
with the safety filter in place: **loop latency 3.27 / 3.53 / 3.94 ms**
(mean/p95/max) over 60 ticks, with the filter clamping injected 9999° spikes on
**21 of 60** ticks and no unsafe target reaching the (simulated) motors. The
terminal capture of this run is `docs/safety_demo.png`.

### A systems note on the client/server split
Serving the trained BC policy over the WebSocket server and querying it from a
local client gave ~110 ms round-trip against ~20 ms of server compute. The
remaining ~90 ms is transport: two uncompressed 480×640×3 frames is ~1.8 MB of
msgpack per request. **On the deployed path the wire, not the model, is the
bottleneck** — JPEG-encoding the frames client-side is the obvious next step.
(Measured on loopback under load; indicative, not a benchmark.)

## 5. Ablations

| Axis | A | B | Result |
|------|---|---|--------|
| Pretraining | BC from scratch | SmolVLA fine-tune | **−66.1% action error** |
| Architecture | BC (single-step MLP) | ACT (CVAE, chunked) | **−58.1% action error** |
| Scale/pretraining | ACT (52M, scratch) | SmolVLA (450M, pretrained) | **−19.3% action error** |
| Text-encoding cache | recompute per tick | cache by string | **2.38× faster inference** |
| Language | no-language control | with-language | **not evaluated** |
| Paraphrase | train phrasings | held-out phrasings | **not evaluated** |
| Data efficiency | 50 demos | 300 demos | **not evaluated** |
| Camera robustness | fixed camera | nudged camera | **not evaluated** |

The ACT → SmolVLA gap (−19.3%) is much smaller than the BC → ACT gap (−58.1%).
On this dataset most of the benefit comes from **action chunking and a temporal
architecture**, not from the 8.7× jump in parameters or from VLA pretraining. A
450M-parameter pretrained VLA bought roughly a fifth off an ACT that trains from
scratch in the same wall-clock time and runs 18× faster.

## 6. Scope decisions and limitations

- **OpenVLA-7B was not trained.** It needs ~24 GB even with QLoRA; the available
  GPU has 12 GB.
- **SmolVLA ran at batch 32, not the upstream recipe's 64** — 64 exhausts 12 GB
  (CUDA failure in `pin_memory`). 32 was measured as the best-throughput size
  that fits (0.82 s/step, 39 samples/s, vs 24 at bs=16 and 16 at bs=8).
- **BC ran 20 epochs, not the config's original 50.** At 20 epochs BC sees 840k
  samples — more than ACT (800k) or SmolVLA (640k) — so the comparison is not
  budget-confounded, and its held-out MSE had clearly flattened (0.0791 at epoch
  14 → 0.0632 at epoch 19). The loop is bound by MP4 decode, not by the GPU.
- **8 held-out episodes is a small sample.** The bootstrap CIs are tight because
  989 frames were scored, but frames within an episode are highly correlated;
  the *effective* sample size is nearer 8 than 989. Treat per-task differences
  cautiously.
- **One scene, one arm, one lighting condition.** Because all four tasks share a
  scene, language grounding was never tested against a *changed* scene.
- **`torchcodec` is unavailable on Windows for torch 2.7**, so video decoding
  falls back to pyav. This dominates BC's training time (GPU util ~1% during
  BC training) and is the main reason a 12M-parameter model took 256 minutes.

## 7. Reproducibility checklist

- [x] Every hyperparameter in a committed YAML (`configs/train_*.yaml`)
- [x] `requirements.txt` fully pinned; `lerobot==0.4.1` recorded in the README
- [x] Seeds fixed (42) across every run in the comparison
- [x] Train/held-out split derived from one function, asserted by unit tests
- [x] Identical held-out frames, order, and metric for all three models
- [x] Training logs committed (`outputs/train_logs/*.log`)
- [x] Eval artifacts committed (`outputs/eval/action_error/*.json`, `*.md`)
- [x] 66 unit/integration tests passing, hardware-free
- [x] W&B run links (project `vla-manipulation`):
      [BC](https://wandb.ai/arjunsood2025-georgia-institute-of-technology/vla-manipulation/runs/wbzll9ra) ·
      [ACT](https://wandb.ai/arjunsood2025-georgia-institute-of-technology/vla-manipulation/runs/wal9nu90) ·
      [SmolVLA](https://wandb.ai/arjunsood2025-georgia-institute-of-technology/vla-manipulation/runs/ldj19ane)
