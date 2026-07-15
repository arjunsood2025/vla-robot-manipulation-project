import random

from vla.data.paraphrases import ParaphraseBank
from vla.eval.trial_spec import TrialSuite

BANK_PATH = "configs/paraphrases.json"
SUITE_PATH = "configs/eval_trials_example.json"


def test_bank_loads_and_separates_pools():
    bank = ParaphraseBank.load(BANK_PATH)
    assert "block_in_bowl" in bank.task_ids()
    task = bank["block_in_bowl"]
    # Train and held-out phrasings must be disjoint — the whole paraphrase-gap
    # metric depends on the held-out sentences never appearing in training.
    assert set(task.train).isdisjoint(set(task.heldout))


def test_sample_returns_from_requested_pool():
    bank = ParaphraseBank.load(BANK_PATH)
    task = bank["block_in_bowl"]
    rng = random.Random(0)
    for _ in range(20):
        assert task.sample("train", rng) in task.train
        assert task.sample("heldout", rng) in task.heldout


def test_trial_suite_loads_and_validates():
    bank = ParaphraseBank.load(BANK_PATH)
    suite = TrialSuite.load(SUITE_PATH)
    suite.validate(set(bank.task_ids()))  # must not raise
    assert suite.trials_per_condition == 20
    assert len(suite.conditions) >= 1


def test_trial_suite_rejects_unknown_task():
    suite = TrialSuite.load(SUITE_PATH)
    try:
        suite.validate({"some_other_task"})
        assert False, "expected validation to fail on unknown task id"
    except ValueError:
        pass
