"""The ModelServer shows a short slate cut from the retrieved candidates,
and training negatives come only from that shown slate."""

import random
from types import SimpleNamespace

import pytest

from ml_system.model_server import ModelServer
from ml_system.training_dataset import TrainingDatasetGenerator
from simulation.events import RecommendationClickedEvent, RecommendationShownEvent


class FakeRetriever:
    def __init__(self):
        self.requests = []

    def retrieve(self, request):
        self.requests.append(request)
        return list(range(1, request.k + 1))


def test_model_server_returns_slate_from_candidates():
    retriever = FakeRetriever()
    server = ModelServer(retriever, num_candidates=100, slate_size=10)

    slate = server.recommend(SimpleNamespace(user_id=1))

    assert retriever.requests[0].k == 100
    assert slate == list(range(1, 11))


def test_model_server_slate_size_is_configurable():
    server = ModelServer(FakeRetriever(), num_candidates=50, slate_size=5)

    assert len(server.recommend(SimpleNamespace(user_id=1))) == 5

    with pytest.raises(ValueError):
        ModelServer(FakeRetriever(), num_candidates=5, slate_size=10)


def test_negatives_only_come_from_shown_slate():
    generator = TrainingDatasetGenerator(
        feature_store=None, negative_samples=50, rng=random.Random(0)
    )
    shown_slate = list(range(1, 11))

    generator.add_event(
        RecommendationShownEvent(
            timestamp=0, user_id=1, recommendation_id=7, movie_ids=shown_slate
        ),
        0,
    )
    generator.add_event(
        RecommendationClickedEvent(
            timestamp=0, user_id=1, recommendation_id=7, movie_id=3
        ),
        1,
    )

    (interaction,) = generator.interactions
    negatives = interaction["negative_movie_ids"]

    assert set(negatives) <= set(shown_slate) - {3}
    assert len(negatives) == len(shown_slate) - 1


def test_simulation_logs_ten_item_slates():
    pytest.importorskip("torch")
    from world.external_data_loader import load_movielens
    from ml_system.build_recommendation_system import build_recommendation_system

    data = load_movielens()
    system = build_recommendation_system(
        movie_objects=data["movie_objects"],
        user_objects=data["user_objects"],
        user_states=data["user_states"],
        movie_states=data["movie_states"],
        seed=7,
    )
    system["sim"].run(30, verbose=False)

    shown = [
        event
        for event in system["datalake"].events
        if isinstance(event, RecommendationShownEvent)
    ]

    assert shown
    assert all(len(event.movie_ids) == 10 for event in shown)
