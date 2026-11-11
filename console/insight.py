"""Where the system loses clicks: a stage-by-stage report.

Uses the World's ground truth (true relevance), which the ML system never
sees, so it is a teaching view for the player ("simulation insight"), not
a model input. For a sample of recently active users it computes the exact
expected CTR (UserSimulator.expected_click_rate, no RNG) of:

- served       what the model server shows now (ranker + re-rankers)
- two_tower    the best 10 of the two-tower candidates
- extra        the best 10 of the extra candidate sources (profile)
- candidates   the best 10 of all candidates the ranker receives
- profile      the candidates ordered by the system's own learned genre
               profile (no ground truth): what a ranker that only trusted
               the profile would serve. served << profile means the ranker
               models badly; profile << candidates means the system does
               not know the users well enough
- oracle       the best 10 of the whole catalog

plus the share of relevant movies among the two-tower candidates, the
extra candidates and the served slate. Retrieval loses oracle - candidates;
ranking loses candidates - served.
"""

import numpy as np

from ml_system.candidate_sources import merge_candidates
from ml_system.ranking import ExplorationReranker, genre_match
from simulation.events import RetrievalRequest
from world.relevance import RELEVANCE_THRESHOLD, relevance_matrix


def stage_report(session, max_users=150, window=300):
    system = session.system
    world = system["world"]
    server = session.model_server
    simulator = world.user_simulator

    users = sorted({e.user_id for e in session._recent_events(window) if hasattr(e, "movie_ids")})
    if not users:
        return None
    step = max(1, len(users) // max_users)
    users = users[::step][:max_users]

    movie_ids = list(world.movie_objects)
    column = {m: i for i, m in enumerate(movie_ids)}
    rel = relevance_matrix([world.user_states[u] for u in users], [world.movie_objects[m] for m in movie_ids])

    totals = {k: 0.0 for k in ("served", "two_tower", "extra", "candidates", "profile", "oracle")}
    relevant = {k: [] for k in ("two_tower", "extra", "served")}
    for row, user_id in enumerate(users):
        user = world.user_objects[user_id]
        request = RetrievalRequest(user_id=user_id, k=server.num_candidates)
        retriever, ranker = session.retriever, server.ranker
        if server.router is not None:
            retriever, ranker = server.router(user_id)
        two_tower = retriever.retrieve(request)
        extras = [source.retrieve(RetrievalRequest(user_id=user_id, k=k)) for source, k in server.extra_sources]
        candidates = merge_candidates(two_tower, extras, server.num_candidates) if extras else list(two_tower)
        ordered = ranker.rank(user, candidates) if ranker is not None else candidates
        # Deterministic re-rankers only: exploration draws from a seeded
        # RNG, and calling it here would change the game.
        for reranker in getattr(server.reranker, "rerankers", [server.reranker] if server.reranker else []):
            if not isinstance(reranker, ExplorationReranker):
                ordered = reranker.rerank(user, ordered)
        served = list(ordered)[:server.slate_size]
        extra = [m for lst in extras for m in lst]

        def best(movies):
            return sorted(set(movies), key=lambda m: (-rel[row, column[m]], m))[:server.slate_size]

        oracle = [movie_ids[j] for j in np.argsort(-rel[row], kind="stable")[:server.slate_size]]
        prefs = (session.retriever.user_features(user_id) or {}).get("genre_preferences") or {}
        by_profile = sorted(candidates, key=lambda m: -genre_match(prefs, world.movie_objects[m].genres))
        totals["profile"] += simulator.expected_click_rate(user_id, by_profile[:server.slate_size])
        ctr = lambda slate: simulator.expected_click_rate(user_id, slate) if slate else 0.0
        totals["served"] += ctr(served)
        totals["two_tower"] += ctr(best(two_tower))
        totals["extra"] += ctr(best(extra))
        totals["candidates"] += ctr(best(candidates))
        totals["oracle"] += ctr(oracle)
        for key, movies in (("two_tower", two_tower), ("extra", extra), ("served", served)):
            if movies:
                relevant[key].append(np.mean([rel[row, column[m]] >= RELEVANCE_THRESHOLD for m in movies]))

    n = len(users)
    report = {k: v / n for k, v in totals.items()}
    report["relevant_share"] = {k: (float(np.mean(v)) if v else None) for k, v in relevant.items()}
    report["users"] = n
    report["tick"] = session.tick
    report["has_extra_sources"] = bool(server.extra_sources)
    report["retrieval_loss"] = report["oracle"] - report["candidates"]
    report["ranking_loss"] = report["candidates"] - report["served"]
    return report
