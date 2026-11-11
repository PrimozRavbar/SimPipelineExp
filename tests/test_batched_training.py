"""Batched two-tower training computes the same loss as the original
one-example-at-a-time loop (up to floating-point rounding)."""

import copy
import random

import pytest

torch = pytest.importorskip("torch")

from ml_system.training import TwoTowerTrainer
from ml_system.two_tower import SimpleItemTower, SimpleUserTower
from world.external_data_loader import GENRE_NAMES


def synthetic_data(n=70, seed=0):
    rng = random.Random(seed)
    genres = GENRE_NAMES[1:]

    def item():
        return {"movie_id": rng.randint(1, 50), "genres": rng.sample(genres, rng.randint(1, 3)),
                "average_rating": rng.uniform(0, 5), "popularity": rng.randint(0, 30)}

    data = []
    for _ in range(n):
        prefs = {g: rng.random() for g in rng.sample(genres, 3)}
        data.append({
            "user_features": {"user_id": rng.randint(1, 30), "genre_preferences": prefs,
                              "average_rating": 3.5, "activity_level": 2, "watch_count": 1,
                              "user_click_counts": 1, "interactions": 3},
            "positive_item_features": item(),
            # Varying negative counts exercise the padding mask.
            "negative_item_features": [item() for _ in range(rng.randint(1, 5))],
        })
    return data


def trainers():
    torch.manual_seed(0)
    user, item = SimpleUserTower(num_users=30), SimpleItemTower(num_movies=50)
    loop = TwoTowerTrainer(copy.deepcopy(user), copy.deepcopy(item), epochs=2, batch_size=16, batched=False)
    fast = TwoTowerTrainer(copy.deepcopy(user), copy.deepcopy(item), epochs=2, batch_size=16, batched=True)
    return loop, fast


def test_batched_training_matches_the_loop():
    data = synthetic_data()
    loop, fast = trainers()
    loop.train(data)
    fast.train(data)

    assert fast.last_losses == pytest.approx(loop.last_losses, rel=1e-4)
    for tower in ("user_tower", "item_tower"):
        a = getattr(loop, tower).state_dict()
        b = getattr(fast, tower).state_dict()
        for key in a:
            assert torch.allclose(a[key], b[key], atol=1e-5), (tower, key)


def test_batched_is_the_default():
    assert TwoTowerTrainer(SimpleUserTower(num_users=3), SimpleItemTower(num_movies=3)).batched
