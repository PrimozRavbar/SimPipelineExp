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

  def recommend(self, user):

      request = RetrievalRequest(
          user_id=user.user_id,
          k=self.num_candidates
      )

      candidates = self.retriever.retrieve(request)

      if self.ranker is not None:
          candidates = self.ranker.rank(
              user,
              candidates
          )

      if self.reranker is not None:
          candidates = self.reranker.rerank(
              user,
              candidates
          )

      return list(candidates)[:self.slate_size]

