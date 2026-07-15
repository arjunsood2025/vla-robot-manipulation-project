# Model Card — SO-101 Language-Conditioned Manipulation Policy

## Model details
- **Family / variant:** `[BC baseline | ACT | SmolVLA | OpenVLA-OFT LoRA]`
- **Input:** front + wrist RGB (640×480), instruction string, 6-DoF joint state.
- **Output:** 6-DoF absolute joint-position target(s) at `[Hz]`; chunked for ACT/VLA.
- **Training data:** `[N]` teleop demos, `[T]` tasks, SO-101, one workspace, controlled lighting.
- **Backbone:** `[e.g. OpenVLA-7B + LoRA r=32, OFT L1 head]`.
- **License:** MIT (this code). Base VLA weights under their own licenses.

## Intended use
Research/education demonstrator for language-conditioned tabletop manipulation on
an SO-101. **Not** for unsupervised operation, human-adjacent tasks, or any
setting outside the trained tabletop domain.

## Metrics (held-out positions + paraphrases, Wilson 95% CI)
See `reports/final_report.md` for the full table. Headline mean success: `[..]`.

## Limitations
- Domain-locked: one arm, one table, one lighting condition, `<10` object types.
- Small eval `n` (20/condition) → wide CIs on any single task.
- Language grounding is imperfect; see the conflicting-instruction probe.
- No depth, no tactile; grasping relies on the wrist RGB view.

## Safety
A hard safety filter (joint-limit + velocity clamps, FK workspace box, gripper
current cap, keyboard e-stop) sits between the policy and the motors. It is a
soft engineering envelope, **not** a certified safety controller. Never run
without the physical e-stop within reach.

## Ethical / responsible-use notes
Trained on self-collected teleoperation data only; no human subjects, no scraped
data. The policy can fail silently (confident wrong grasp) — always supervise.
