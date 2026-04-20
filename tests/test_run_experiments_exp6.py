"""Tests for Exp 6 (domain transfer) paths in run_experiments."""

from argparse import Namespace

import src.run_experiments as rexp


class _DummyPipeline:
    def predict(self, texts):
        # Deterministic and valid CEFR ids.
        import numpy as np

        return np.array([0 for _ in texts])


def test_run_exp6_returns_metrics_and_track(monkeypatch):
    def fake_load_dataset(dataset_name, text_column, label_column):
        if dataset_name == "train_ds":
            return ["a", "a", "b", "c"], [0, 0, 1, 1]
        return ["x", "y", "z"], [0, 1, 1]

    monkeypatch.setattr("src.data_utils.load_dataset", fake_load_dataset)
    monkeypatch.setattr("src.baseline_tfidf.train_baseline", lambda texts, labels: _DummyPipeline())

    result = rexp.run_exp6(
        train_dataset="train_ds",
        eval_dataset="eval_ds",
        track="essay",
    )

    assert result.name.startswith("Exp 6")
    assert result.track == "essay"
    assert "train=train_ds" in result.note
    assert "eval=eval_ds" in result.note
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0
    assert -1.0 <= result.qwk <= 1.0


def test_main_exp6_uses_dataset_as_default_train_eval(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        rexp,
        "parse_args",
        lambda: Namespace(
            task="sentence",
            exps=[6],
            dataset="UniversalCEFR/cefr_sp_en",
            train_dataset=None,
            eval_dataset=None,
            text_column="text",
            label_column="label",
            epochs=1,
            batch_size=8,
            seed=42,
            data_dir=None,
            save_results=None,
        ),
    )
    monkeypatch.setattr(rexp, "set_seed", lambda seed: None)

    def fake_run_exp6(**kwargs):
        calls.update(kwargs)
        return rexp.ExperimentResult(name="Exp 6", track=kwargs["track"])

    monkeypatch.setattr(rexp, "run_exp6", fake_run_exp6)
    monkeypatch.setattr(rexp, "print_comparison_table", lambda results: None)

    rexp.main()

    assert calls["train_dataset"] == "UniversalCEFR/cefr_sp_en"
    assert calls["eval_dataset"] == "UniversalCEFR/cefr_sp_en"
    assert calls["track"] == "sentence"
