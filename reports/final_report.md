# Language-Conditioned VLA Manipulation — Final Report

> Fill the bracketed `[…]` numbers from your own runs. The structure, tables, and
> analysis prompts are the deliverable; the numbers are what you measure. Every
> rate is reported with a Wilson 95% CI (the evaluation harness prints these).

## 1. Problem & setup

We train a single policy that maps **(front + wrist RGB, natural-language
instruction, 6-DoF joint state) → next joint-target action** for an SO-101 arm
performing tabletop pick/place, on a 6×4 (10 cm cell) workspace. Objects: colored
1″ blocks, pinch bowls, nesting cups, ping-pong balls.

- **Robot:** SO-101 follower + SO-101 leader (teleop). 6× Feetech STS3215, 30 Hz.
- **Cameras:** front (overhead 45°) + wrist, both 640×480 @ 30 fps.
- **Data:** `[N]` teleoperated demonstrations across `[T]` tasks, `[k]` training
  phrasings per task, held-out phrasings/cells reserved for evaluation.
- **Compute:** `[GPU]`; LeRobot commit `[hash]`; seeds fixed at 42.

## 2. Models compared

| Model | Params | Action head | Control freq | Notes |
|-------|--------|-------------|--------------|-------|
| BC baseline (ours) | ~15M trainable | MLP, single step | `[Hz]` | CLIP text (frozen) + ResNet-18 + MLP |
| ACT | ~80M | CVAE, chunk=100 | `[Hz]` | temporal ensembling at inference |
| SmolVLA | 450M | flow/diffusion | `[Hz]` | fine-tuned from `smolvla_base` |
| OpenVLA-OFT | 7B (LoRA r=32) | L1 regression, chunk=8 | `[Hz]` | the headline VLA fine-tune |

## 3. Headline results

Success rate on held-out object positions + held-out paraphrases, `[n]` trials
per cell, Wilson 95% CI.

| Task | BC | ACT | SmolVLA | OpenVLA |
|------|----|-----|---------|---------|
| Block → bowl | `[..]` | `[..]` | `[..]` | `[..]` |
| Ball → cup | `[..]` | `[..]` | `[..]` | `[..]` |
| Left-of relation | `[..]` | `[..]` | `[..]` | `[..]` |
| Stack two | `[..]` | `[..]` | `[..]` | `[..]` |
| Distractor (type) | `[..]` | `[..]` | `[..]` | `[..]` |
| **Mean** | `[..]` | `[..]` | `[..]` | `[..]` |

## 4. Ablations

| Ablation | Condition A | Condition B | Δ success | Takeaway |
|----------|-------------|-------------|-----------|----------|
| Pretraining | BC from scratch | OpenVLA fine-tune | `[+X%]` | Does generalist pretraining help? |
| Data efficiency | 50 demos | 300 demos | `[+X%]` | Where does the success curve knee? |
| Perception | RGB only | RGB + wrist | `[+X%]` | Second view mostly helps grasping |
| Language | no-language ctrl | with-language | `[+X%]` | Does the instruction actually drive behavior? |
| Paraphrase | train phrasings | held-out phrasings | `[−X%]` | The generalization gap |
| Camera robustness | fixed camera | ±5 cm nudge | `[−X%]` | Sensitivity to calibration drift |

**Language ablation detail (the one skeptics ask about):** we retrain with the
instruction embedding zeroed. If success barely drops, the policy is solving the
task from vision + a fixed scene prior, *not* from language — so we also report
success on a **conflicting-instruction** probe (two valid targets on the table,
instruction names one). A policy that ignores language scores ~50% (chance)
there; a grounded one scores `[..]`.

## 5. Latency

End-to-end (image capture → action applied), measured client-side over `[n]`
ticks.

| Model | mean | p95 | max | Achieved control freq |
|-------|------|-----|-----|-----------------------|
| BC | `[..]` ms | `[..]` | `[..]` | `[..]` Hz |
| ACT | `[..]` ms | `[..]` | `[..]` | `[..]` Hz |
| OpenVLA (token) | `[..]` ms | `[..]` | `[..]` | ~5 Hz |
| OpenVLA-OFT (chunk+parallel) | `[..]` ms | `[..]` | `[..]` | 25+ Hz |

## 6. Failure analysis

| Failure type | Observed rate | Root cause (hypothesis) | Fix attempted | Result |
|--------------|---------------|-------------------------|---------------|--------|
| Picks wrong object | `[..]` | weak language grounding | add distractor demos | `[+X%]` distractor success |
| Misses grasp | `[..]` | depth/pose ambiguity from one view | add wrist camera | `[+X%]` grasp success |
| Slow / jerky | `[..]` | large VLA latency, chunk seams | OFT chunking + temporal ensemble | `[X]×` faster loop |
| Overfits positions | `[..]` | fixed training layout | seeded position randomization | held-out cell `[+X%]` |
| Out-of-workspace lunge | `[..]` | policy extrapolation | safety filter hold-on-violation | 0 unsafe commands reached motors |

## 7. Limitations & safety

- `[n]=20` trials/condition → ±~20 pt CIs; treat single-task deltas cautiously,
  lean on the mean across tasks.
- Single tabletop, single arm, controlled lighting; no claims beyond this domain.
- Safety filter is a **soft** envelope (FK approximation + velocity/limit clamps),
  not a certified safety system. The keyboard e-stop is the real backstop.

## 8. Reproducibility checklist

- [ ] LeRobot commit hash recorded, `requirements.txt` pinned
- [ ] Dataset versions tagged on the Hub (v1, v2, …), never overwritten
- [ ] Seeds fixed (42) across every run in a comparison
- [ ] Eval suite JSON committed; identical suite run against every model
- [ ] W&B run links pasted here: `[…]`
