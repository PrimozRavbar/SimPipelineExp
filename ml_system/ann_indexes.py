"""Alternatives to HNSW behind the same ANNIndex interface (swappable).

- ExactIndex: brute-force cosine search over every vector. Perfect recall;
  its cost grows with the catalog (every query touches every vector).
- IVFIndex: inverted file (as in FAISS IndexIVFFlat): k-means splits the
  vectors into `nlist` cells; a query searches only the `nprobe` cells
  whose centroids are nearest. Cheap; recall depends on nprobe / nlist.

Both expose `nodes` (item id -> node with `.embedding` and unit `.unit`),
like HNSWIndex, so feature logging and the index self-test work with any of
them. `INDEX_TYPES` describes each type for the console: its class, a
relative query cost (vectors touched per query, for the simulated latency
and the bill) and its running price.
"""

import math
import random

import numpy as np

from ml_system.ann import ANNIndex, HNSWIndex, unit


class _Node:
    __slots__ = ("item_id", "embedding", "unit")

    def __init__(self, item_id, embedding):
        self.item_id = item_id
        self.embedding = embedding
        self.unit = unit(embedding)


class ExactIndex(ANNIndex):

    def __init__(self, seed=0):
        self.nodes = {}
        self._ids = []
        self._matrix = None

    def build(self, item_embeddings):
        for item_id, embedding in item_embeddings.items():
            self.nodes[item_id] = _Node(item_id, embedding)
        self._ids = list(self.nodes)
        self._matrix = np.array([self.nodes[i].unit for i in self._ids], dtype=float)

    def search(self, query_embedding, k):
        if not self._ids:
            return []
        scores = self._matrix @ np.array(unit(query_embedding), dtype=float)
        order = np.lexsort((np.arange(len(self._ids)), -scores))[:k]
        return [self._ids[i] for i in order]

    def vectors_per_query(self):
        return len(self._ids)


class IVFIndex(ANNIndex):

    def __init__(self, nlist=32, nprobe=4, iterations=10, seed=0):
        self.nlist = nlist
        self.nprobe = nprobe
        self.iterations = iterations
        self.seed = seed
        self.nodes = {}
        self._ids = []
        self._matrix = None
        self.centroids = None
        self.cells = []

    def build(self, item_embeddings):
        for item_id, embedding in item_embeddings.items():
            self.nodes[item_id] = _Node(item_id, embedding)
        self._ids = list(self.nodes)
        if not self._ids:
            return
        X = np.array([self.nodes[i].unit for i in self._ids], dtype=float)
        self._matrix = X
        n = min(self.nlist, len(self._ids))
        rng = random.Random("ivf-%s" % self.seed)
        centroids = X[sorted(rng.sample(range(len(X)), n))].copy()
        for _ in range(self.iterations):            # spherical k-means
            assign = np.argmax(X @ centroids.T, axis=1)
            for c in range(n):
                members = X[assign == c]
                if len(members):
                    v = members.sum(axis=0)
                    norm = np.linalg.norm(v)
                    centroids[c] = v / norm if norm else centroids[c]
        self.centroids = centroids
        assign = np.argmax(X @ centroids.T, axis=1)
        self.cells = [np.where(assign == c)[0] for c in range(n)]

    def search(self, query_embedding, k):
        if not self._ids:
            return []
        q = np.array(unit(query_embedding), dtype=float)
        probe = np.argsort(-(self.centroids @ q), kind="stable")[:self.nprobe]
        rows = np.concatenate([self.cells[c] for c in probe]) if len(probe) else np.array([], dtype=int)
        if not len(rows):
            return []
        scores = self._matrix[rows] @ q
        order = np.lexsort((rows, -scores))[:k]
        return [self._ids[rows[i]] for i in order]

    def vectors_per_query(self):
        if not self._ids:
            return 0
        sizes = sorted((len(c) for c in self.cells), reverse=True)
        return len(self.centroids) + sum(sizes[:self.nprobe])


def hnsw_vectors_per_query(index):
    """Distance computations per query, roughly: the ef_search beam on the
    bottom layer plus M links per layer on the way down."""
    n = max(len(getattr(index, "nodes", {})), 2)
    return int(index.ef_search + index.M * math.log2(n))


INDEX_TYPES = {
    "hnsw": {"name": "HNSW (graph)", "price_per_tick": 0.6,
             "about": "Graph search: fast, high recall (~96%), memory-heavy."},
    "ivf": {"name": "IVF (FAISS-style inverted file)", "price_per_tick": 0.3,
            "about": "Clusters, searches the nprobe nearest: cheapest; recall depends on nprobe."},
    "exact": {"name": "Exact (brute force)", "price_per_tick": 1.5,
              "about": "Every vector, every query: perfect recall, the most compute."},
}


def make_index(kind, seed=0, nprobe=4):
    if kind == "hnsw":
        return HNSWIndex(seed=seed)
    if kind == "ivf":
        return IVFIndex(seed=seed, nprobe=nprobe)
    if kind == "exact":
        return ExactIndex(seed=seed)
    raise ValueError("unknown index type %r (one of %s)" % (kind, sorted(INDEX_TYPES)))


def index_kind(index):
    base = index
    while hasattr(base, "inner"):
        base = base.inner
    return {HNSWIndex: "hnsw", IVFIndex: "ivf", ExactIndex: "exact"}.get(type(base), "other")


def index_embeddings(index):
    base = index
    while hasattr(base, "inner"):
        base = base.inner
    return {item_id: node.embedding for item_id, node in base.nodes.items()}


def vectors_per_query(index):
    base = index
    while hasattr(base, "inner"):
        base = base.inner
    if isinstance(base, HNSWIndex):
        return hnsw_vectors_per_query(base)
    return base.vectors_per_query()
