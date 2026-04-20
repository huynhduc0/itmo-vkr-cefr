"""Unit tests for Exp 11-14 wrappers/dispatch without GPU training."""

from argparse import Namespace

import src.run_experiments as rexp


def test_run_exp11_delegates_to_run_exp2(monkeypatch):
    captured = {}

    def fake_run_exp2(**kwargs):
        captured.update(kwargs)
        return rexp.ExperimentResult(name="ok", track=kwargs["track"])

    monkeypatch.setattr(rexp, "run_exp2", fake_run_exp2)

    result = rexp.run_exp11([], [], [], [], [], [], track="sentence", num_epochs=2, batch_size=4, seed=7)

    assert result.track == "sentence"
    assert captured["model_name"] == "microsoft/deberta-v3-base"
    assert captured["num_epochs"] == 2
    assert captured["batch_size"] == 4
    assert captured["seed"] == 7


def test_run_exp12_delegates_to_run_exp3(monkeypatch):
    captured = {}

    def fake_run_exp3(**kwargs):
        captured.update(kwargs)
        return rexp.ExperimentResult(name="ok", track=kwargs["track"])

    monkeypatch.setattr(rexp, "run_exp3", fake_run_exp3)

    result = rexp.run_exp12([], [], [], [], [], [], track="essay", num_epochs=3, batch_size=2, seed=9)

    assert result.track == "essay"
    assert captured["model_name"] == "microsoft/deberta-v3-base"
    assert captured["num_epochs"] == 3
    assert captured["batch_size"] == 2
    assert captured["seed"] == 9


def test_run_exp14_averages_three_exp4_runs(monkeypatch):
    calls = []

    def fake_run_exp4(**kwargs):
        calls.append(kwargs["seed"])
        seed = kwargs["seed"]
        return rexp.ExperimentResult(
            name="exp4",
            track=kwargs["track"],
            accuracy=0.5 + 0.01 * seed,
            macro_f1=0.4 + 0.01 * seed,
            qwk=0.3 + 0.01 * seed,
            latency=0.02,
        )

    monkeypatch.setattr(rexp, "run_exp4", fake_run_exp4)

    result = rexp.run_exp14([], [], [], [], [], [], track="sentence", seed=10)

    assert calls == [10, 11, 12]
    assert result.name.startswith("Exp 14")
    assert result.track == "sentence"
    assert result.note == "mean over 3 seeds"
    assert result.latency == 0.02


def test_main_dispatches_exp11_to_exp14(monkeypatch):
    called = {"11": False, "12": False, "13": False, "14": False}

    monkeypatch.setattr(
        rexp,
        "parse_args",
        lambda: Namespace(
            task="sentence",
            exps=[11, 12, 13, 14],
            dataset="d",
            train_dataset=None,
            eval_dataset=None,
            text_column="text",
            label_column="label",
            epochs=1,
            batch_size=8,
            seed=42,
            data_dir="dummy",
            save_results=None,
        ),
    )
    monkeypatch.setattr(rexp, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        rexp,
        "_load_splits_from_jsonl",
        lambda data_dir, task: ((["t"], [0]), (["v"], [0]), (["e"], [0])),
    )

    monkeypatch.setattr(rexp, "run_exp11", lambda *args, **kwargs: called.__setitem__("11", True) or rexp.ExperimentResult(name="11", track="sentence"))
    monkeypatch.setattr(rexp, "run_exp12", lambda *args, **kwargs: called.__setitem__("12", True) or rexp.ExperimentResult(name="12", track="sentence"))
    monkeypatch.setattr(rexp, "run_exp13", lambda *args, **kwargs: called.__setitem__("13", True) or rexp.ExperimentResult(name="13", track="sentence"))
    monkeypatch.setattr(rexp, "run_exp14", lambda *args, **kwargs: called.__setitem__("14", True) or rexp.ExperimentResult(name="14", track="sentence"))
    monkeypatch.setattr(rexp, "print_comparison_table", lambda results: None)

    rexp.main()

    assert all(called.values())
