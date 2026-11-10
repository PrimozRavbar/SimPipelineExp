from abc import ABC, abstractmethod
from simulation.service import Service
from simulation.events import RetrievalRequest

class RetrievalService(Service, ABC):

    @abstractmethod
    def retrieve(self, request: RetrievalRequest):
        pass


class ModelServer(Service):

  def __init__(
      self,
      retriever,
      ranker=None,
      reranker=None,
      num_candidates=100,
      slate_size=10
  ):
      super().__init__()

      if slate_size > num_candidates:
          raise ValueError("slate_size cannot exceed num_candidates")

      self.retriever = retriever
      self.ranker = ranker
      self.reranker = reranker

      # Retrieve a wide candidate set, then show the user only the top
      # `slate_size` of it. Only the slate is logged as shown, so only
      # items the user could actually see become training negatives.
      self.num_candidates = num_candidates
      self.slate_size = slate_size

      # Degraded mode: while `retriever_down` is set (an outage), requests
      # are served by `fallback` (e.g. PopularityRetrieval). Optional
      # `router(user_id) -> (retriever, ranker)` picks a model per user
      # (canary / A-B tests). All off by default.
      self.fallback = None
      self.retriever_down = False
      self.router = None
      self.requests = 0
      self.fallback_requests = 0

  def recommend(self, user):

      request = RetrievalRequest(
          user_id=user.user_id,
          k=self.num_candidates
      )

      retriever, ranker = self.retriever, self.ranker
      if self.router is not None:
          retriever, ranker = self.router(user.user_id)

      self.requests += 1
      if self.retriever_down and self.fallback is not None:
          self.fallback_requests += 1
          retriever, ranker = self.fallback, None

      candidates = retriever.retrieve(request)

      if ranker is not None:
          candidates = ranker.rank(
              user,
              candidates
          )

      if self.reranker is not None:
          candidates = self.reranker.rerank(
              user,
              candidates
          )

      return list(candidates)[:self.slate_size]

