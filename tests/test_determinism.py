"""Fast checks that seeded components are reproducible. No PyTorch needed."""

import random
from types import SimpleNamespace

from ml_system.ann import HNSWIndex
from world.external_data_loader import load_movielens
from world.user_simulator import UserSimulator


def random_embeddings(count=200, dim=16, seed=0):
    rng = random.Random(seed)
    embeddings = {}

    for item_id in range(1, count + 1):
        vector = [rng.gauss(0, 1) for _ in range(dim)]
        norm = sum(x * x for x in vector) ** 0.5
        embeddings[item_id] = [x / norm for x in vector]

    return embeddings


def graph_of(index):
    return (
        index.entry_point,
        index.max_level,
        {
            item_id: {
                layer: list(neighbors)
                for layer, neighbors in node.neighbors.items()
            }
            for item_id, node in index.nodes.items()
        },
    )


def test_hnsw_same_seed_same_graph():
    embeddings = random_embeddings()

    first = HNSWIndex(seed=3)
    first.build(embeddings)

    second = HNSWIndex(seed=3)
    second.build(embeddings)

    assert graph_of(first) == graph_of(second)


def test_user_simulator_same_seed_same_reactions():
    data = load_movielens()
    world = SimpleNamespace(
        user_states=data["user_states"],
        movie_objects=data["movie_objects"],
    )
    sim = SimpleNamespace(clock=SimpleNamespace(now=0))

    users = list(data["user_objects"].values())[:50]
    movie_ids = list(data["movie_objects"])[:10]

    def run(seed):
        simulator = UserSimulator(world, rng=random.Random(seed))
        reactions = []

        for recommendation_id, user in enumerate(users):
            for event in simulator.react(sim, user, recommendation_id, movie_ids):
                reactions.append(repr(event))

        return reactions

    assert run(11) == run(11)
    assert run(11) != run(12)
