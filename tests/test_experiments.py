"""Canary releases / A-B tests (ml_system/experiments.py, console canary)."""

import pytest

from ml_system.experiments import bucket, in_treatment, two_proportion_test


def test_bucket_is_stable_and_roughly_uniform():
    assert bucket(5, 1) == bucket(5, 1)
    shares = sum(in_treatment(u, 3, 20) for u in range(1, 2001)) / 2000
    assert 0.17 < shares < 0.23
    # A new salt reshuffles: arm overlap between two experiments ~ 50%.
    same = sum((bucket(u, 1) < 50) == (bucket(u, 2) < 50) for u in range(1, 2001)) / 2000
    assert 0.4 < same < 0.6


def test_two_proportion_test():
    result = two_proportion_test(300, 1000, 200, 1000)
    assert result["diff"] == pytest.approx(-0.1)
    assert result["z"] < -1.96 and result["high"] < 0
    assert two_proportion_test(0, 0, 1, 10) is None


torch = pytest.importorskip("torch")
from console.session import SimSession  # noqa: E402


def canary_session():
    s = SimSession(preset="plain", seed=3, users_per_tick=4, log_metrics=False,
                   settings={"deploy_delay": 10, "epochs": 1, "spark_every": 0,
                             "canary_share": 50, "canary_ticks": 20})
    s.step(30)
    s.submit("retrain")
    s.step(11)                                   # training done; canary starts at tick 40
    return s


def test_canary_splits_traffic_then_promotes():
    s = canary_session()
    assert s.canary is not None and s.serving_version == 0
    assert s.model_server.router is not None
    s.step(18)                                   # canary served ticks 41..59 so far
    counts = s.canary["experiment"].counts
    clicks = sum(1 for e in s.system["datalake"].events
                 if type(e).__name__ == "RecommendationClickedEvent" and 41 <= e.timestamp <= 59)
    assert counts["control"][0] + counts["treatment"][0] == 19 * 4
    assert counts["control"][1] + counts["treatment"][1] == clicks > 0
    s.step(1)                                    # ends after tick 60
    assert s.canary is None and s.model_server.router is None
    stats = s.versions[1].info["canary"]
    assert stats["result"] in ("promoted", "rolled back")
    if stats["result"] == "promoted":
        assert s.serving_version == 1
    s.close()


def test_losing_canary_rolls_back_with_an_incident():
    s = canary_session()
    s.canary["experiment"].counts = {"control": [1000, 300], "treatment": [1000, 100]}
    s.step(20)
    assert s.serving_version == 0 and s.canary is None
    assert s.versions[1].info["canary"]["result"] == "rolled back"
    incident = s.incidents_view()[0]
    assert incident["rule"] == "canary" and incident["status"] == "Mitigated"
    s.close()


def test_manual_promote_and_no_retrain_during_canary():
    s = canary_session()
    blocked = s.submit("retrain")
    s.submit("promote_canary")
    s.step(1)
    assert blocked["status"] == "rejected"
    assert s.serving_version == 1 and s.canary is None
    s.close()
