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
    """User tower: ID embedding + genre profile + activity counts -> MLP.

    `use_id_embedding` / `use_activity_features` switch input groups off
    (their inputs are fed as zeros, so the layer shapes and checkpoints
    stay the same). Activity counts (average rating, activity level,
    watch/click/interaction counts) only grow over time, so at serving
    time they are always larger than anything seen in the point-in-time
    training examples; with a few events per user they mostly add noise.
    build_recommendation_system turns them off by default.
    """

    def __init__(
        self,
        num_users,
        embedding_dim=32,
        use_id_embedding=True,
        use_activity_features=True
    ):
        super().__init__()

        self.use_id_embedding = use_id_embedding
        self.use_activity_features = use_activity_features

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

        if not self.use_id_embedding:
            user_id_embedding = torch.zeros_like(user_id_embedding)

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

        activity = [
            average_rating,
            normalize_count(user_features.get("activity_level", 0)),
            normalize_count(user_features.get("watch_count", 0)),
            normalize_count(user_features.get("user_click_counts", 0)),
            normalize_count(user_features.get("interactions", 0)),
        ]

        if not self.use_activity_features:
            activity = [0.0] * len(activity)

        embedding.extend(activity)

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
    """Item tower: ID embedding + genres + average rating + popularity.

    Input groups can be switched off the same way as in SimpleUserTower.
    `use_rating_feature=False` drops the item's average rating, which is
    noise while simulated ratings are random.
    """

    def __init__(
        self,
        num_movies,
        embedding_dim=32,
        use_id_embedding=True,
        use_rating_feature=True,
        use_popularity_feature=True
    ):
        super().__init__()

        self.use_id_embedding = use_id_embedding
        self.use_rating_feature = use_rating_feature
        self.use_popularity_feature = use_popularity_feature

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

        if not self.use_id_embedding:
            movie_id_embedding = torch.zeros_like(movie_id_embedding)

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

        embedding.append(average_rating if self.use_rating_feature else 0.0)
        embedding.append(popularity if self.use_popularity_feature else 0.0)

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

# Input groups the production model uses (build_recommendation_system and
# RetrievalEvaluator defaults). See the tower docstrings for why activity
# counts and the item average rating are off.
DEFAULT_USER_TOWER_KWARGS = {"use_id_embedding": False, "use_activity_features": False}
DEFAULT_ITEM_TOWER_KWARGS = {"use_id_embedding": False, "use_rating_feature": False}


# Value a user has before any interaction, identical to the offline
# pipeline's initial user state, so serving and training agree.
USER_FEATURE_DEFAULTS = {
    "genre_preferences": {},
    "average_rating": 0,
    "activity_level": 0,
    "watch_count": 0,
    "user_click_counts": 0,
    "interactions": 0,
}


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


    def user_features(self, user_id):
        """Online user features, defined exactly as the offline pipeline
        defines them for training (see OfflineFeaturePipeline).

        Each feature comes from Redis (Flink, fresh) if present, else from
        the offline feature store (last Spark/retrain run), else the same
        default a brand-new user has in the training data: an empty genre
        profile and zero counts. Serving must not invent values that
        training never saw (it used to send a uniform profile over a
        hard-coded genre list and a 3.5 average rating to cold-start users,
        and `interactions` was never updated online).
        """
        offline = self.feature_store.get_user_features(user_id) or {}

        features = {"user_id": user_id}

        for name, default in USER_FEATURE_DEFAULTS.items():
            value = self.redis.get(name, user_id)

            if value is None:
                value = offline.get(name, default)

            features[name] = value

        return features


    def retrieve(self, request):

        user_embedding = self.user_tower.embed(
            self.user_features(request.user_id)
        )

        return self.ann_index.search(
            user_embedding,
            request.k
        )
