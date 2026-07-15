# Demo videos

Large binaries are **not** committed to git (see `.gitignore`). Record these,
upload them (YouTube unlisted, or a GitHub Release asset), and link them from the
top-level `README.md`.

Expected files:

| File | What it shows |
|------|---------------|
| `demo_successes.mp4` | 4–6 clean rollouts on **unseen** instructions and object positions, split-screen with the front + wrist camera feeds. |
| `failure_cases.mp4`  | Representative failures (wrong-object grasp, missed grasp, out-of-workspace hold) — the honesty reel that makes the success reel credible. |

**How to produce them:** the evaluation harness already logs every trial. Screen-
record the operator station during an eval run (OBS Studio, free), or save the
camera streams to disk during rollouts and stitch the best takes in any editor.
Overlay the instruction text and the success/failure label on each clip.
