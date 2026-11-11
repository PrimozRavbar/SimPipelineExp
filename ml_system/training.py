import math
import random

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
        batch_size=32,
        batched=True,
        logq_correction=False,
        random_negatives=0,
        graded=False,
        seed=0
    ):

        self.user_tower = user_tower
        self.item_tower = item_tower

        self.epochs = epochs
        self.temperature = temperature
        self.batch_size = batch_size

        # batched=True runs each tower once per batch (the same loss, much
        # faster on CPU); False keeps the original one-example-at-a-time
        # loop. Results differ only by floating-point rounding.
        self.batched = batched
        self.last_losses = []

        # Retrieval-quality options (batched path only; all off by default):
        # - logq_correction: subtract log(item frequency among positives)
        #   from the positive and in-batch logits (sampling-bias correction,
        #   Yi et al. 2019), so popular items aren't over-penalised as
        #   in-batch negatives.
        # - random_negatives: per example, this many random catalog items
        #   from `item_pool` (list of item feature dicts) as extra negatives.
        # - graded: weight each example's loss by example["weight"]
        #   (TrainingDatasetGenerator(graded=True) sets it from what
        #   happened after the click).
        self.logq_correction = logq_correction
        self.random_negatives = random_negatives
        self.graded = graded
        self.item_pool = None
        self.seed = seed
        self.train_calls = 0

        self.optimizer = torch.optim.Adam(
            list(self.user_tower.parameters()) +
            list(self.item_tower.parameters()),
            lr=learning_rate
        )


    def train(self, training_data):

        if not self.batched and (self.logq_correction or self.random_negatives or self.graded):
            raise ValueError("logq_correction, random_negatives and graded need batched=True")

        if self.batched:
            return self._train_batched(training_data)

        self.user_tower.train()
        self.item_tower.train()
        self.last_losses = []

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
            self.last_losses.append(total_loss / batch_count)

        return (
            self.user_tower,
            self.item_tower
        )


    def _train_batched(self, training_data):
        """Same loss as the loop above, computed per batch with tensors:
        for example i the logits are [its positive, its sampled negatives,
        the other examples' positives (in-batch negatives)] / temperature,
        target 0, averaged over the batch. Examples with fewer sampled
        negatives are padded and masked out."""

        self.user_tower.train()
        self.item_tower.train()
        self.last_losses = []

        # Seeded per training call, so the same retrains give the same model.
        self.train_calls += 1
        rng = random.Random("negatives-%s-%d" % (self.seed, self.train_calls))
        log_q = None
        if self.logq_correction:
            frequency = {}
            for e in training_data:
                m = e["positive_item_features"].get("movie_id")
                frequency[m] = frequency.get(m, 0) + 1
            log_q = {m: math.log(c / len(training_data)) for m, c in frequency.items()}
        pool = self.item_pool if self.random_negatives else None
        if self.random_negatives and not pool:
            raise ValueError("random_negatives needs trainer.item_pool")

        for epoch in range(self.epochs):

            total_loss = 0
            batch_count = 0

            for start in range(0, len(training_data), self.batch_size):

                batch = training_data[start:start + self.batch_size]
                size = len(batch)

                users = F.normalize(self.user_tower.forward(torch.stack([
                    self.user_tower.encode_features(e["user_features"]) for e in batch
                ])), p=2, dim=1)

                positives = F.normalize(self.item_tower.forward(torch.stack([
                    self.item_tower.encode_features(e["positive_item_features"]) for e in batch
                ])), p=2, dim=1)

                counts = [len(e["negative_item_features"]) for e in batch]
                width = max(counts)
                flat = [f for e in batch for f in e["negative_item_features"]]
                negatives = F.normalize(self.item_tower.forward(torch.stack([
                    self.item_tower.encode_features(f) for f in flat
                ])), p=2, dim=1)

                # (size, width) sampled-negative scores, padded with -inf.
                padded = torch.zeros(size, width, negatives.shape[1])
                mask = torch.zeros(size, width, dtype=torch.bool)
                offset = 0
                for i, count in enumerate(counts):
                    padded[i, :count] = negatives[offset:offset + count]
                    mask[i, :count] = True
                    offset += count
                sampled = torch.einsum("bd,bwd->bw", users, padded)
                sampled = sampled.masked_fill(~mask, float("-inf"))

                pair = (users @ positives.T) / self.temperature  # (size, size)
                if log_q is not None:
                    correction = torch.tensor([log_q[e["positive_item_features"].get("movie_id")]
                                               for e in batch], dtype=pair.dtype)
                    pair = pair - correction.unsqueeze(0)
                positive = pair.diagonal().unsqueeze(1)          # (size, 1)
                off_diagonal = ~torch.eye(size, dtype=torch.bool)
                in_batch = pair[off_diagonal].view(size, size - 1)

                parts = [positive, sampled / self.temperature, in_batch]
                if pool:
                    drawn = [pool[rng.randrange(len(pool))] for _ in range(size * self.random_negatives)]
                    rand = F.normalize(self.item_tower.forward(torch.stack([
                        self.item_tower.encode_features(f) for f in drawn
                    ])), p=2, dim=1).view(size, self.random_negatives, -1)
                    parts.append(torch.einsum("bd,brd->br", users, rand) / self.temperature)

                logits = torch.cat(parts, dim=1)
                target = torch.zeros(size, dtype=torch.long)
                if self.graded:
                    weights = torch.tensor([float(e.get("weight", 1.0)) for e in batch])
                    per_example = F.cross_entropy(logits, target, reduction="none")
                    loss = (per_example * weights).sum() / weights.sum()
                else:
                    loss = F.cross_entropy(logits, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            print("epoch:", epoch + 1, "loss:", total_loss / batch_count)
            self.last_losses.append(total_loss / batch_count)

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
