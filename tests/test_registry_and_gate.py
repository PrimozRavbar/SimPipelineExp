"""Model bundles on disk (ml_system/model_registry.py) and the console's
offline evaluation gate."""

import pytest

torch = pytest.importorskip("torch")

from console.session import SimSession
from ml_system.model_registry import load_bundle


def test_gate_scores_both_models_and_blocks_a_worse_one(tmp_path):
    s = SimSession(seed=3, users_per_tick=4, settings={
        "deploy_delay": 10, "epochs": 1, "spark_every": 0, "eval_gate": True,
        "holdout_pct": 20, "gate_tolerance_pct": 0}, runs_dir=str(tmp_path), run_id="gate")
    s.step(60)
    s.submit("retrain")
    s.step(12)
    gate = s.versions[1].info["gate"]
    assert gate["holdout_clicks"] > 0
    assert 0 <= gate["recall_at_20"] <= 1 and gate["production_version"] == 0
    assert gate["passed"] == (gate["recall_at_20"] >= gate["production_recall_at_20"])
    assert s.serving_version == (1 if gate["passed"] else 0)
    if not gate["passed"]:
        assert s.models()["versions"][0]["stage"] == "Rejected"

    # Export round trip: same embeddings, same top-k from the rebuilt index.
    s.submit("export", version=1)
    s.step(1)
    bundle = load_bundle(tmp_path / "gate" / "models" / "v1", index_seed=s.seed * 1000 + 1)
    original = s.versions[1]
    features = s.retriever.user_features(5)
    assert bundle["user_tower"].embed(features) == pytest.approx(original.user_tower.embed(features))
    query = original.user_tower.embed(features)
    assert bundle["index"].search(query, 10) == original.index.search(query, 10)
    assert bundle["metadata"]["version"] == 1
    s.close()
