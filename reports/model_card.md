# Model Card — SO-101 Language-Conditioned Manipulation Policies

> **Scope:** Three policies were trained (BC baseline, ACT, SmolVLA
> fine-tune) and evaluated **offline only**.

## Model details
- **Family / variants:** from-scratch BC baseline (12.4M trainable), ACT
  (52M, CVAE, chunk=100), SmolVLA (450M, fine-tuned from `lerobot/smolvla_base`).
- **Input:** front + wrist RGB (640×480), instruction string, 6-DoF joint state.
- **Output:** 6-DoF absolute joint-position targets; single-step for BC,
  100-action chunks for ACT/SmolVLA.
- **Training data:** [`youliangtan/so101-table-cleanup`](https://huggingface.co/datasets/youliangtan/so101-table-cleanup)
  — a **public, third-party** SO-101 dataset. 80 episodes / 46,963 frames /
  4 language tasks, recorded on someone else's arm in their workspace and
  lighting. **Not collected by the author of this repository.**
- **Train/held-out split:** 72 / 8 episodes, split by episode (never by frame),
  seed 42, identical for all three models (`vla/data/splits.py`).
- **License:** MIT (this code). The dataset and base VLA weights are under their
  own licenses.

## Intended use
A research/education demonstrator for the *engineering* of a language-conditioned
manipulation stack: data pipeline, training ladder, safety filter, client/server
inference, and evaluation design. **Not** a deployable controller, and **not**
evidence that any of these policies can perform a manipulation task.

## Evaluation
Held-out **action-prediction error**: mean absolute difference between the
policy's action and the demonstrator's, on episodes no model trained on,
open-loop and single-step, in physical units. Full table in
`reports/final_report.md`; figure in `docs/results.png`.

**What this metric cannot tell you.** It is measured open-loop against recorded
demonstrations: the policy is always shown a *demonstrator's* state, never a
state its own earlier actions produced. Real rollouts compound error, so a low
score here does not imply the policy would complete the task. A policy can also
track the average trajectory closely while never actually closing the gripper.
Treat these numbers as a *relative* ranking of the three rungs under identical
conditions — nothing more.

**Not evaluated (requires hardware):** task success rate, paraphrase
generalization gap, distractor robustness, unseen-object-position
generalization, the language-ablation probe, and closed-loop latency to a real
arm. The harness for all of these exists (`scripts/evaluate.py`,
`configs/eval_trials_example.json`) and is exercised in `--dry-run` mode, but
**dry-run labels are a seeded Bernoulli draw, not measurements** — they must
never be reported as results.

## Limitations
- **No closed-loop evaluation of any kind.** See above.
- Domain-locked: one arm, one table, one lighting condition, one scene layout.
- All four tasks share a single scene; the instruction is the only thing that
  distinguishes them. That makes the language signal clean, but it also means
  language grounding was never tested against a *changed* scene.
- 8 held-out episodes is a small sample; per-task numbers are noisy.
- SmolVLA was trained at batch 32 rather than the upstream recipe's 64 (12 GB
  VRAM limit), and OpenVLA-7B was not trained at all — it does not fit on the
  available hardware. The ladder is therefore BC → ACT → SmolVLA.
- No depth, no tactile sensing; grasping would rely on the wrist RGB view.

## Safety
A hard safety filter (joint-limit + velocity clamps, FK workspace box, gripper
current cap, keyboard e-stop) sits between the policy and the motors, and is
exercised offline in `scripts/demo_offline.py`. It is a soft engineering
envelope, **not** a certified safety controller, and it has only ever been
tested against a simulated robot. Never run without a physical e-stop in reach.

## Ethical / responsible-use notes
Training used a publicly published third-party teleoperation dataset, credited
above; no human subjects and no scraped data. These policies can fail silently —
a confident, wrong grasp looks identical to a correct one until it executes —
so any hardware use must be supervised.
