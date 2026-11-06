from abc import ABC, abstractmethod




from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from world.external_data_loader import GENRE_NAMES

genre_names = GENRE_NAMES


class UserTower(ABC):

    @abstractmethod
    def embed(self, user_features):
        pass



class SimpleUserTower(nn.Module, UserTower):

    def __init__(self, num_users, embedding_dim=32):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_users + 1,
            16
        )

        input_dim = 16 + len(genre_names) + 5

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )


    def encode_features(self, user_features):

        user_id = user_features["user_id"]

        user_id_embedding = self.user_embedding(
            torch.tensor(
                user_id,
                dtype=torch.long
            )
        )

        embedding = []

        genre_preferences = user_features["genre_preferences"]

        for genre in genre_names:
            embedding.append(
                float(genre_preferences.get(genre, 0.0))
            )

        average_rating = max(
            0.0,
            min(
                5.0,
                float(user_features.get("average_rating", 0) or 0)
            )
        ) / 5.0

        def normalize_count(value):
            value = max(0.0, float(value or 0))
            return float(
                torch.tanh(
                    torch.log1p(
                        torch.tensor(value)
                    )
                )
            )

        embedding.append(average_rating)
        embedding.append(
            normalize_count(
                user_features.get("activity_level", 0)
            )
        )
        embedding.append(
            normalize_count(
                user_features.get("watch_count", 0)
            )
        )
        embedding.append(
            normalize_count(
                user_features.get("user_click_counts", 0)
            )
        )
        embedding.append(
            normalize_count(
                user_features.get("interactions", 0)
            )
        )

        metadata_embedding = torch.tensor(
            embedding,
            dtype=torch.float32
        )

        return torch.cat(
            [
                user_id_embedding,
                metadata_embedding
            ]
        )


    def forward(self, x):

        return self.network(x)


    def encode(self, user_features):

        x = self.encode_features(
            user_features
        )

        return self.forward(x)


    def embed(self, user_features):

        with torch.no_grad():

            embedding = self.encode(
                user_features
            )

            return F.normalize(
                embedding,
                p=2,
                dim=0
            ).tolist()



class ItemTower(ABC):

    @abstractmethod
    def embed(self, movie_features):
        pass



class SimpleItemTower(nn.Module, ItemTower):

    def __init__(
        self,
        num_movies,
        embedding_dim=32
    ):
        super().__init__()

        self.movie_embedding = nn.Embedding(
            num_movies + 1,
            16
        )

        input_dim = 16 + len(genre_names) + 2

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )


    def encode_features(self, movie_features):

        movie_id = movie_features["movie_id"]

        movie_id_embedding = self.movie_embedding(
            torch.tensor(
                movie_id,
                dtype=torch.long
            )
        )

        embedding = []

        movie_genres = set(
            movie_features["genres"]
        )

        for genre in genre_names:
            embedding.append(
                1.0 if genre in movie_genres else 0.0
            )

        average_rating = max(
            0.0,
            min(
                5.0,
                float(
                    movie_features.get(
                        "average_rating",
                        0
                    ) or 0
                )
            )
        ) / 5.0

        popularity = max(
            0.0,
            float(
                movie_features.get(
                    "popularity",
                    0
                ) or 0
            )
        )

        popularity = float(
            torch.tanh(
                torch.log1p(
                    torch.tensor(popularity)
                )
            )
        )

        embedding.append(average_rating)
        embedding.append(popularity)

        metadata_embedding = torch.tensor(
            embedding,
            dtype=torch.float32
        )

        return torch.cat(
            [
                movie_id_embedding,
                metadata_embedding
            ]
        )


    def forward(self, x):

        return self.network(x)


    def encode(self, movie_features):

        x = self.encode_features(
            movie_features
        )

        return self.forward(x)


    def embed(self, movie_features):

        with torch.no_grad():

            embedding = self.encode(
                movie_features
            )

            return F.normalize(
                embedding,
                p=2,
                dim=0
            ).tolist()

from simulation.events import RetrievalRequest
from ml_system.model_server import RetrievalService

class TwoTowerRetrieval(RetrievalService):

    def __init__(
        self,
        user_tower,
        item_tower,
        ann_index,
        redis,
        feature_store
    ):
        super().__init__()

        self.user_tower = user_tower
        self.item_tower = item_tower
        self.ann_index = ann_index
        self.redis = redis
        self.feature_store = feature_store


    def build_index(self, world):

        item_embeddings = {}

        for movie_id, movie in world.movie_objects.items():

            state = world.movie_states[movie_id]

            features = {
                "movie_id": movie_id,
                "genres": movie.genres,
                "average_rating": state.average_rating,
                "popularity": state.num_ratings
            }

            item_embeddings[movie_id] = (
                self.item_tower.embed(features)
            )

        self.ann_index.build(item_embeddings)


    def retrieve(self, request):

        user_id = request.user_id


        genre_preferences = self.redis.get(
            "genre_preferences",
            user_id
        )

        average_rating = self.redis.get(
            "average_rating",
            user_id
        )

        activity_level = self.redis.get(
            "activity_level",
            user_id
        )

        watch_count = self.redis.get(
            "watch_count",
            user_id
        )

        user_click_counts = self.redis.get(
            "user_click_counts",
            user_id
        )

        interactions = self.redis.get(
            "interactions",
            user_id
        )


        # Cold start bootstrap
        if genre_preferences is None:

            user_features = self.feature_store.user_features.get(
                user_id
            )

            if user_features is None:

                genre_names = [
                    "Action",
                    "Adventure",
                    "Animation",
                    "Comedy",
                    "Crime",
                    "Drama",
                    "Fantasy",
                    "Horror",
                    "Romance",
                    "Sci-Fi",
                    "Thriller",
                    "Documentary",
                    "Family",
                    "Mystery",
                    "War",
                    "Western",
                    "Music"
                ]

                weight = 1.0 / len(genre_names)

                user_features = {
                    "genre_preferences": {
                        genre: weight
                        for genre in genre_names
                    },
                    "average_rating": 3.5,
                    "activity_level": 0,
                    "watch_count": 0,
                    "user_click_counts": 0,
                    "interactions": 0
                }

                self.feature_store.write_user_features(
                    user_id,
                    user_features
                )


            genre_preferences = user_features.get(
                "genre_preferences"
            )

            average_rating = user_features.get(
                "average_rating",
                3.5
            )

            activity_level = user_features.get(
                "activity_level",
                0
            )

            watch_count = user_features.get(
                "watch_count",
                0
            )

            user_click_counts = user_features.get(
                "user_click_counts",
                0
            )

            interactions = user_features.get(
                "interactions",
                0
            )


            # Materialize complete online feature state

            for feature, value in {
                "genre_preferences": genre_preferences,
                "average_rating": average_rating,
                "activity_level": activity_level,
                "watch_count": watch_count,
                "user_click_counts": user_click_counts,
                "interactions": interactions
            }.items():

                self.redis.receive({
                    "feature": feature,
                    "user_id": user_id,
                    "value": value
                })


        # Recover missing online features from offline store

        if (
            average_rating is None or
            activity_level is None or
            watch_count is None or
            user_click_counts is None or
            interactions is None
        ):

            offline = self.feature_store.user_features.get(
                user_id,
                {}
            )


            average_rating = (
                average_rating
                if average_rating is not None
                else offline.get("average_rating", 3.5)
            )

            activity_level = (
                activity_level
                if activity_level is not None
                else offline.get("activity_level", 0)
            )

            watch_count = (
                watch_count
                if watch_count is not None
                else offline.get("watch_count", 0)
            )

            user_click_counts = (
                user_click_counts
                if user_click_counts is not None
                else offline.get("user_click_counts", 0)
            )

            interactions = (
                interactions
                if interactions is not None
                else offline.get("interactions", 0)
            )


        user_features = {
            "user_id": user_id,
            "genre_preferences": genre_preferences,
            "average_rating": average_rating,
            "activity_level": activity_level,
            "watch_count": watch_count,
            "user_click_counts": user_click_counts,
            "interactions": interactions
        }


        user_embedding = self.user_tower.embed(
            user_features
        )


        return self.ann_index.search(
            user_embedding,
            request.k
        )