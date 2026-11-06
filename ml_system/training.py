import torch
import torch.nn.functional as F


class TwoTowerTrainer:

    def __init__(
        self,
        user_tower,
        item_tower,
        learning_rate=0.001,
        epochs=5,
        temperature=0.1,
        batch_size=32
    ):

        self.user_tower = user_tower
        self.item_tower = item_tower

        self.epochs = epochs
        self.temperature = temperature
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(
            list(self.user_tower.parameters()) +
            list(self.item_tower.parameters()),
            lr=learning_rate
        )


    def train(self, training_data):

        self.user_tower.train()
        self.item_tower.train()

        for epoch in range(self.epochs):

            total_loss = 0
            batch_count = 0

            for start in range(
                0,
                len(training_data),
                self.batch_size
            ):

                batch = training_data[
                    start:start + self.batch_size
                ]

                user_embeddings = torch.stack([
                    F.normalize(
                        self.user_tower.encode(
                            example["user_features"]
                        ),
                        p=2,
                        dim=0
                    )
                    for example in batch
                ])

                positive_embeddings = torch.stack([
                    F.normalize(
                        self.item_tower.encode(
                            example["positive_item_features"]
                        ),
                        p=2,
                        dim=0
                    )
                    for example in batch
                ])

                positive_scores = (
                    user_embeddings
                    @ positive_embeddings.T
                )

                losses = []

                for i, example in enumerate(batch):

                    positive_score = positive_scores[i, i]

                    sampled_negative_embeddings = torch.stack([
                        F.normalize(
                            self.item_tower.encode(features),
                            p=2,
                            dim=0
                        )
                        for features in
                        example["negative_item_features"]
                    ])

                    sampled_negative_scores = (
                        user_embeddings[i]
                        @ sampled_negative_embeddings.T
                    )

                    in_batch_negative_scores = torch.cat([
                        positive_scores[i, :i],
                        positive_scores[i, i + 1:]
                    ])

                    scores = torch.cat([
                        positive_score.unsqueeze(0),
                        sampled_negative_scores,
                        in_batch_negative_scores
                    ])

                    logits = (
                        scores /
                        self.temperature
                    )

                    target = torch.zeros(
                        1,
                        dtype=torch.long
                    )

                    loss = F.cross_entropy(
                        logits.unsqueeze(0),
                        target
                    )

                    losses.append(loss)

                loss = torch.stack(losses).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            print(
                "epoch:",
                epoch + 1,
                "loss:",
                total_loss / batch_count
            )

        return (
            self.user_tower,
            self.item_tower
        )


class EmbeddingGenerator:

    def __init__(self, feature_store):
        self.feature_store = feature_store

    def generate(self, item_tower):

        embeddings = {}

        for movie_id in self.feature_store.item_features:

            features = self.feature_store.get_item_features(
                movie_id
            )

            embeddings[movie_id] = (
                item_tower.embed(features)
            )

        return embeddings
