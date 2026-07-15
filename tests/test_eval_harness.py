import random

from vla.data.paraphrases import ParaphraseBank
from vla.eval.evaluate import run_condition, render_report, compute_paraphrase_gap
from vla.eval.trial_spec import Condition


def _bank():
    return ParaphraseBank.load("configs/paraphrases.json")


def test_dry_run_condition_counts_sum_to_trials():
    cond = Condition(
        condition_id="train_phrasing_seen",
        task_id="block_in_bowl",
        phrasing_source="train",
        layout={"red block": "C2", "green bowl": "D3"},
        seed=0,
    )
    result, records = run_condition(cond, n_trials=20, bank=_bank(), rollout=None,
                                    dry_run=True, rng=random.Random(0))
    assert result.n == 20
    assert result.successes + result.partials + result.failures == 20
    assert len(records) == 20
    # Dry-run instructions must come from the train pool for a train condition.
    train_phrasings = set(_bank()["block_in_bowl"].train)
    assert all(r.instruction in train_phrasings for r in records)


def test_dry_run_is_reproducible():
    cond = Condition("c", "block_in_bowl", "train", {"red block": "C2"}, seed=1)
    r1, _ = run_condition(cond, 20, _bank(), None, True, random.Random(42))
    r2, _ = run_condition(cond, 20, _bank(), None, True, random.Random(42))
    assert (r1.successes, r1.partials, r1.failures) == (r2.successes, r2.partials, r2.failures)


def test_report_renders_markdown_table():
    cond = Condition("train_phrasing_x", "block_in_bowl", "train", {"red block": "C2"}, seed=0)
    result, _ = run_condition(cond, 20, _bank(), None, True, random.Random(0))
    md = render_report([result])
    assert md.startswith("| Condition |")
    assert "train_phrasing_x" in md


def test_paraphrase_gap_computed_when_pair_present():
    bank = _bank()
    train_cond = Condition("train_phrasing_a", "block_in_bowl", "train", {}, seed=0)
    held_cond = Condition("heldout_phrasing_a", "block_in_bowl", "heldout", {}, seed=1)
    t, _ = run_condition(train_cond, 20, bank, None, True, random.Random(0))
    h, _ = run_condition(held_cond, 20, bank, None, True, random.Random(1))
    gap = compute_paraphrase_gap([t, h])
    assert gap is not None
