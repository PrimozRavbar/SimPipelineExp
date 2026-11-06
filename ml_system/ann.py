import math
import random
import heapq
from abc import ABC, abstractmethod

def cosine_distance(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return 1 - (dot / (norm_a * norm_b))

class ANNIndex(ABC):

    @abstractmethod
    def build(self, item_embeddings):
        pass

    @abstractmethod
    def search(self, query_embedding, k):
        pass


class HNSWNode:

    def __init__(self, item_id, embedding, level):
        self.item_id = item_id
        self.embedding = embedding
        self.level = level

        # neighbors[level] = list of connected node ids
        self.neighbors = {
            i: []
            for i in range(level + 1)
        }

import random
import heapq


class HNSWIndex(ANNIndex):

    #def __init__(self, M=16, ef_construction=200):

    def __init__(
        self,
        M=16,
        ef_construction=200,
        ef_search=200
    ):

        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self.nodes = {}
        self.entry_point = None
        self.max_level = -1

        self.level_mult = 1 / math.log(M)

    def distance_to_node(self, query_embedding, node_id):

        return cosine_distance(
            query_embedding,
            self.nodes[node_id].embedding
        )

    def assign_level(self):
        return int(
            -math.log(random.random()) * self.level_mult
        )

    def insert(self, item_id, embedding):

        level = self.assign_level()

        node = HNSWNode(
            item_id,
            embedding,
            level
        )

        self.nodes[item_id] = node

        if self.entry_point is None:
            self.entry_point = item_id
            self.max_level = level
            return

    def cosine_distance(a, b):

        dot = sum(x * y for x, y in zip(a, b))

        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5

        return 1 - (dot / (norm_a * norm_b))


    def select_neighbors(self, candidates):

        candidates.sort(
            key=lambda x: x[1]
        )

        return candidates[:self.M]

    def insert(self, item_id, embedding):

        level = self.assign_level()

        node = HNSWNode(
            item_id,
            embedding,
            level
        )

        self.nodes[item_id] = node

        if self.entry_point is None:
            self.entry_point = item_id
            self.max_level = level
            return

        # Connect node from its highest available layer down to layer 0
        for layer in range(min(level, self.max_level), -1, -1):

            candidates = []

            for other_id, other_node in self.nodes.items():

                if other_id == item_id:
                    continue

                if layer <= other_node.level:

                    distance = cosine_distance(
                        embedding,
                        other_node.embedding
                    )

                    candidates.append(
                        (other_id, distance)
                    )

            neighbors = self.select_neighbors(candidates)

            node.neighbors[layer] = [
                neighbor_id
                for neighbor_id, _ in neighbors
            ]

            # Create bidirectional connections
            for neighbor_id in node.neighbors[layer]:

                neighbor = self.nodes[neighbor_id]

                neighbor.neighbors[layer].append(
                    item_id
                )

                self.prune_neighbors(
                    neighbor,
                    layer
                )

        # Update entry point if this node has a higher level
        if level > self.max_level:

            self.entry_point = item_id
            self.max_level = level



    def build(self, item_embeddings):

        for item_id, embedding in item_embeddings.items():

            self.insert(
                item_id,
                embedding
            )


    def prune_neighbors(self, node, layer):

        neighbors = node.neighbors[layer]

        if len(neighbors) <= self.M:
            return

        candidates = []

        for neighbor_id in neighbors:

            neighbor = self.nodes[neighbor_id]

            distance = cosine_distance(
                node.embedding,
                neighbor.embedding
            )

            candidates.append(
                (neighbor_id, distance)
            )

        selected = self.select_neighbors(candidates)

        node.neighbors[layer] = [
            neighbor_id
            for neighbor_id, _ in selected
        ]


    def search_layer_greedy(self, query_embedding, entry_point, layer):

        current = entry_point

        current_distance = self.distance_to_node(
            query_embedding,
            current
        )

        changed = True

        while changed:

            changed = False

            node = self.nodes[current]

            for neighbor_id in node.neighbors.get(layer, []):

                distance = self.distance_to_node(
                    query_embedding,
                    neighbor_id
                )

                if distance < current_distance:

                    current = neighbor_id
                    current_distance = distance
                    changed = True

        return current



    def search_layer(self, query_embedding, entry_point, ef):

        visited = set()

        # min heap: closest candidates to explore
        candidates = []

        # max heap: best results found
        results = []

        distance = self.distance_to_node(
            query_embedding,
            entry_point
        )

        heapq.heappush(
            candidates,
            (distance, entry_point)
        )

        heapq.heappush(
            results,
            (-distance, entry_point)
        )

        visited.add(entry_point)


        while candidates:

            current_distance, current_id = heapq.heappop(
                candidates
            )

            # Furthest result is better than current candidate:
            # stop expanding
            if len(results) >= ef:

                worst_distance = -results[0][0]

                if current_distance > worst_distance:
                    break


            node = self.nodes[current_id]

            for neighbor_id in node.neighbors.get(0, []):

                if neighbor_id in visited:
                    continue

                visited.add(neighbor_id)

                distance = self.distance_to_node(
                    query_embedding,
                    neighbor_id
                )

                heapq.heappush(
                    candidates,
                    (distance, neighbor_id)
                )

                heapq.heappush(
                    results,
                    (-distance, neighbor_id)
                )

                if len(results) > ef:
                    heapq.heappop(results)


        return [
            node_id
            for _, node_id in sorted(
                [
                    (-d, node_id)
                    for d, node_id in results
                ]
            )
        ]

    def search(self, query_embedding, k):

        # Start from entry point
        current = self.entry_point

        # Descend upper layers
        for layer in range(self.max_level, 0, -1):

            current = self.search_layer_greedy(
                query_embedding,
                current,
                layer
            )

        # Layer 0 efSearch
        candidates = self.search_layer(
            query_embedding,
            current,
            ef=self.ef_search
        )

        return candidates[:k]