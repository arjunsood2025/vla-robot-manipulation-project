# Held-out action-prediction error (Lane A)

Dataset: `youliangtan/so101-table-cleanup` — 72 train / 8 held-out episodes.
Scored 989 held-out frames (every 5th), open-loop, single step, physical units.

| Model | Frames | MAE | 95% CI | Median AE | RMSE |
|---|---|---|---|---|---|
| bc | 989 | 5.1281 | [4.9858, 5.2760] | 4.8082 | 7.2435 |
| act | 989 | 2.1508 | [2.0903, 2.2138] | 2.0046 | 3.0685 |
| smolvla | 989 | 1.7357 | [1.6729, 1.7972] | 1.5387 | 2.6972 |

Per-task MAE:

| Model | Grab markers and place into pen holder | Grab pens and place into pen holder | Grab scissor and place into pen holder | Grab tapes and place into pen holder |
|---|---|---|---|---|
| bc | 4.5302 | 5.1967 | 6.2459 | 5.0822 |
| act | 2.2124 | 1.6764 | 2.5499 | 2.0544 |
| smolvla | 1.7969 | 1.3754 | 1.9298 | 1.7323 |

> Open-loop action error is **not** a task-success rate. It compares how closely each policy reproduces held-out demonstrations; it does not establish that any of them would complete the task on hardware.
