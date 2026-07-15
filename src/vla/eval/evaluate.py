"""Operator-in-the-loop evaluation harness.

For each condition in a trial suite we run N trials. Each trial:
  1. print the object layout for the operator to set up (grid cells),
  2. sample an instruction from the train or held-out phrasing pool,
  3. run the policy rollout on the robot (via the inference client), and
  4. record the operator's success / partial / failure keypress.

Everything is logged to a CSV (and optionally W&B), then aggregated into a
per-condition table with Wilson 95% confidence intervals. The same suite is run
unchanged against every model so the comparison is apples-to-apples.

A ``--dry-run`` mode replaces the robot rollout and the operator prompt with a
seeded Bernoulli draw, so the harness logic itself can be exercised on any
machine with no hardware (this is what the unit tests and a laptop demo use).
"""
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from vla.data.paraphrases import ParaphraseBank
from vla.eval.metrics import ConditionResult, paraphrase_gap, wilson_interval
from vla.eval.trial_spec import Condition, TrialSuite

# outcome label -> canonical bucket
_VALID = {"s": "success", "p": "partial", "f": "failure"}

# A rollout function takes (instruction, layout, distractors, seed) and returns
# nothing meaningful — it just drives the robot. The operator judges the result.
RolloutFn = Callable[[str, dict, list, int], None]


@dataclass
class TrialRecord:
    condition_id: str
    task_id: str
    trial_index: int
    phrasing_source: str
    instruction: str
    outcome: str
    seed: int


def _operator_label(instruction: str) -> str:
    """Blocking keypress prompt. Returns 'success' | 'partial' | 'failure'."""
    while True:
        key = input(f"    Outcome for [{instruction}]  (s)uccess / (p)artial / (f)ailure: ").strip().lower()
        if key in _VALID:
            return _VALID[key]
        print("    Please press s, p or f.")


def _dry_run_label(instruction: str, rng: random.Random, p_success: float = 0.6) -> str:
    """Simulated outcome for hardware-free testing/demo."""
    r = rng.random()
    if r < p_success:
        return "success"
    if r < p_success + 0.15:
        return "partial"
    return "failure"


def run_condition(
    condition: Condition,
    n_trials: int,
    bank: ParaphraseBank,
    rollout: RolloutFn | None,
    dry_run: bool,
    rng: random.Random,
) -> tuple[ConditionResult, list[TrialRecord]]:
    task = bank[condition.task_id]
    records: list[TrialRecord] = []
    counts = {"success": 0, "partial": 0, "failure": 0}

    print(f"\n=== Condition '{condition.condition_id}' ({n_trials} trials) ===")
    for i in range(n_trials):
        trial_rng = random.Random(condition.seed * 10_000 + i)
        instruction = task.sample(condition.phrasing_source, trial_rng)

        if not dry_run:
            print(f"  Trial {i + 1}/{n_trials}: set up layout {condition.layout}"
                  + (f" + distractors {condition.distractors}" if condition.distractors else ""))
            input("    Press ENTER when the scene is ready...")
            if rollout is not None:
                rollout(instruction, condition.layout, condition.distractors, condition.seed + i)
            outcome = _operator_label(instruction)
        else:
            outcome = _dry_run_label(instruction, rng)

        counts[outcome] += 1
        records.append(TrialRecord(
            condition_id=condition.condition_id,
            task_id=condition.task_id,
            trial_index=i,
            phrasing_source=condition.phrasing_source,
            instruction=instruction,
            outcome=outcome,
            seed=condition.seed + i,
        ))

    result = ConditionResult(
        condition_id=condition.condition_id,
        n=n_trials,
        successes=counts["success"],
        partials=counts["partial"],
        failures=counts["failure"],
    )
    print(f"  -> success {result.success_rate}")
    return result, records


def write_trial_csv(records: list[TrialRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition_id", "task_id", "trial_index", "phrasing_source",
                         "instruction", "outcome", "seed"])
        for r in records:
            writer.writerow([r.condition_id, r.task_id, r.trial_index, r.phrasing_source,
                             r.instruction, r.outcome, r.seed])


def render_report(results: list[ConditionResult]) -> str:
    """Markdown table of per-condition success with Wilson 95% CIs."""
    lines = ["| Condition | n | Success (95% CI) | Partial+Success (95% CI) |",
             "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r.condition_id} | {r.n} | {r.success_rate} | {r.partial_or_success_rate} |")
    return "\n".join(lines)


def compute_paraphrase_gap(results: list[ConditionResult]) -> float | None:
    """If the suite contains matched train_/heldout_ conditions on the same
    layout, report the paraphrase gap between the first such pair."""
    by_id = {r.condition_id: r for r in results}
    train = next((r for cid, r in by_id.items() if cid.startswith("train_phrasing")), None)
    heldout = next((r for cid, r in by_id.items() if cid.startswith("heldout_phrasing")), None)
    if train and heldout:
        return paraphrase_gap(train.success_rate.point, heldout.success_rate.point)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the manipulation evaluation suite.")
    parser.add_argument("--suite", required=True, help="Path to a trial-spec JSON.")
    parser.add_argument("--paraphrases", default="configs/paraphrases.json")
    parser.add_argument("--server-uri", default="ws://localhost:8000",
                        help="Policy server URI for the real rollout.")
    parser.add_argument("--out-dir", default="outputs/eval")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate outcomes; no robot or operator needed.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    bank = ParaphraseBank.load(args.paraphrases)
    suite = TrialSuite.load(args.suite)
    suite.validate(set(bank.task_ids()))

    rollout: RolloutFn | None = None
    if not args.dry_run:
        # Lazy import so --dry-run has zero hardware/network dependencies.
        from vla.inference.client import RobotClient

        client = RobotClient(server_uri=args.server_uri)
        rollout = client.run_episode

    rng = random.Random(args.seed)
    all_results: list[ConditionResult] = []
    all_records: list[TrialRecord] = []
    for condition in suite.conditions:
        result, records = run_condition(
            condition, suite.trials_per_condition, bank, rollout, args.dry_run, rng
        )
        all_results.append(result)
        all_records.extend(records)

    out_dir = Path(args.out_dir) / suite.suite_name
    write_trial_csv(all_records, out_dir / "trials.csv")
    report = render_report(all_results)
    (out_dir / "report.md").write_text(report + "\n", encoding="utf-8")

    print("\n" + report)
    gap = compute_paraphrase_gap(all_results)
    if gap is not None:
        print(f"\nParaphrase gap (train - heldout success): {100 * gap:+.0f} points")
    print(f"\nWrote {out_dir/'trials.csv'} and {out_dir/'report.md'}")


if __name__ == "__main__":
    main()
