import torch

from ml_system.two_tower import SimpleUserTower, SimpleItemTower
from ml_system.ann import HNSWIndex


def load_two_tower_checkpoint(
    checkpoint_path,
    movie_objects,
    movie_states,
    retriever
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    user_tower = SimpleUserTower(
        num_users=checkpoint["user_num_embeddings"] - 1,
        embedding_dim=checkpoint["tower_output_dim"]
    )

    item_tower = SimpleItemTower(
        num_movies=checkpoint["item_num_embeddings"] - 1,
        embedding_dim=checkpoint["tower_output_dim"]
    )

    user_tower.load_state_dict(
        checkpoint["user_tower_state_dict"]
    )

    item_tower.load_state_dict(
        checkpoint["item_tower_state_dict"]
    )

    item_embeddings = {}

    for movie_id, movie in movie_objects.items():

        features = {
            "movie_id": movie_id,
            "genres": movie.genres,
            "average_rating": movie_states[movie_id].average_rating,
            "popularity": movie_states[movie_id].num_ratings
        }

        item_embeddings[movie_id] = (
            item_tower.embed(features)
        )

    hnsw_index = HNSWIndex()
    hnsw_index.build(item_embeddings)

    retriever.user_tower = user_tower
    retriever.item_tower = item_tower
    retriever.ann_index = hnsw_index

    return checkpoint
