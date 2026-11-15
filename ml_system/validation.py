"""Checks a retrain workflow runs before training and before deploying.

Data validation (TFDV / Great Expectations style) looks at the training set
the workflow is about to train on and compares it with the last training
set that passed: too few examples, a click rate far from the usual, or many
users with empty profiles (a broken feature job) stop the run before any
GPU time is spent.

Model validation looks at what training produced: a loss that is not a
number or that went up (diverged), weights or embeddings that are not
finite, or item embeddings that all collapsed into one point.

Both read only ML-system data (never World ground truth) and use no
randomness. A check is {"name", "value", "limit", "ok", "about"}.
"""

import math

# Data checks
MIN_EXAMPLES = 1            # absolute floor; the relative checks below do the work
MIN_ROWS_PCT = 60            # at least 60% of the last good training set
MAX_EMPTY_PROFILE_JUMP = 15  # points more users with empty profiles than last time
CLICK_RATE_FLOOR = 0.5       # click rate (labels) at least half the last good run's

# Model checks
MAX_LOSS_RATIO = 1.2         # the last epoch's loss at most 1.2x the first's
MAX_LOSS_VS_PRODUCTION = 1.06  # on the same examples, at most 1.06x the serving model's loss
MIN_EMBEDDING_SPREAD = 1e-4  # std of item embeddings across movies


def _check(name, value, limit, ok, about):
    return {"name": name, "value": value, "limit": limit, "ok": bool(ok), "about": about}


def training_data_stats(examples, events, feature_store, since_tick=None):
    """Summary of a training set: its size, the click rate of the events
    behind it, and the share of clicks since `since_tick` (the last good
    run; None = all) by returning users whose profile, as of the click, was
    empty. A broken feature job corrupts the newest data first, so the
    newest slice is what is checked. Only returning users count (a click in
    a later tick than the user's first event): a new user's first clicks
    legitimately see an empty profile, and the lake even stores a tick's
    clicks before that tick's onboarding answers (one file per topic)."""
    shown = clicks = recent = empty = 0
    first_tick = {}
    for order, event in enumerate(events):
        first_tick.setdefault(event.user_id, event.timestamp)
        name = type(event).__name__
        if name == "RecommendationShownEvent":
            shown += 1
        elif name == "RecommendationClickedEvent":
            clicks += 1
            if (since_tick is None or event.timestamp > since_tick) and first_tick[event.user_id] < event.timestamp:
                recent += 1
                features = feature_store.get_user_features_as_of(event.user_id, event.timestamp, order)
                if not (features or {}).get("genre_preferences"):
                    empty += 1
    return {
        "examples": examples,
        "click_rate": round(clicks / shown, 4) if shown else 0.0,
        "new_clicks": recent,
        "empty_profile_pct": round(100.0 * empty / recent, 1) if recent else 0.0,
        "through_tick": events[-1].timestamp if events else 0,
    }


def validate_data(stats, reference=None):
    """Checks of `stats` (training_data_stats) against the last training set
    that passed (`reference`, None on the first run)."""
    checks = [_check("examples", stats["examples"], MIN_EXAMPLES, stats["examples"] >= MIN_EXAMPLES,
                     "enough examples to train on")]
    if reference:
        floor = reference["examples"] * MIN_ROWS_PCT / 100.0
        checks.append(_check("examples vs last run", stats["examples"], round(floor),
                             stats["examples"] >= floor,
                             "at least %d%% of the last good training set (%d)" % (MIN_ROWS_PCT, reference["examples"])))
        limit = reference["empty_profile_pct"] + MAX_EMPTY_PROFILE_JUMP
        checks.append(_check("empty profiles in new data (%)", stats["empty_profile_pct"], round(limit, 1),
                             stats["empty_profile_pct"] <= limit,
                             "users with no genre profile: at most %d points more than last time (%.1f%%)"
                             % (MAX_EMPTY_PROFILE_JUMP, reference["empty_profile_pct"])))
        low = reference["click_rate"] * CLICK_RATE_FLOOR
        if reference["click_rate"]:
            checks.append(_check("click rate", stats["click_rate"], round(low, 4), stats["click_rate"] >= low,
                                 "labels: at least x%.1f the last good run's (%.1f%%); rising is fine"
                                 % (CLICK_RATE_FLOOR, reference["click_rate"] * 100)))
    return checks


def validate_model(losses, embeddings, eval_loss=None, production_eval_loss=None):
    """Checks of a trained model: its per-epoch losses, its item embeddings
    ({movie_id: tensor or list}) and, if given, its loss and the production
    model's loss on the same sample of the new training set
    (training.evaluation_loss). A bad learning rate rarely makes this
    model's loss blow up (its embeddings are normalised): it trains a worse
    model, which only a comparison on the same data shows. (Comparing
    final training losses across retrains does not work: a model trained
    on little data reaches a lower loss on it.)"""
    checks = []
    finite = bool(losses) and all(math.isfinite(x) for x in losses)
    checks.append(_check("loss is a number", losses[-1] if losses else None, "finite", finite,
                         "training produced a finite loss"))
    if finite:
        ratio = losses[-1] / losses[0] if losses[0] else 1.0
        checks.append(_check("loss did not diverge", round(ratio, 3), MAX_LOSS_RATIO, ratio <= MAX_LOSS_RATIO,
                             "last epoch's loss / first epoch's loss"))
    if eval_loss is not None and production_eval_loss:
        vs = eval_loss / production_eval_loss if math.isfinite(eval_loss) else float("inf")
        checks.append(_check("loss vs production model", round(vs, 3), MAX_LOSS_VS_PRODUCTION,
                             vs <= MAX_LOSS_VS_PRODUCTION,
                             "on the same new examples: this model's loss / the serving model's (%.3f)"
                             % production_eval_loss))
    vectors = [list(map(float, v)) for v in embeddings.values()]
    ok = bool(vectors) and all(math.isfinite(x) for v in vectors for x in v)
    checks.append(_check("embeddings are finite", len(vectors), "all", ok, "no NaN or inf in item embeddings"))
    if ok:
        dims = len(vectors[0])
        n = len(vectors)
        spread = 0.0
        for d in range(dims):
            mean = sum(v[d] for v in vectors) / n
            spread += sum((v[d] - mean) ** 2 for v in vectors) / n
        spread = math.sqrt(spread / dims)
        checks.append(_check("embeddings not collapsed", round(spread, 6), MIN_EMBEDDING_SPREAD,
                             spread >= MIN_EMBEDDING_SPREAD, "item embeddings must differ between movies"))
    return checks


def passed(checks):
    return all(c["ok"] for c in checks)
