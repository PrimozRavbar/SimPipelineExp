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
      reranker=None
  ):
      super().__init__()

      self.retriever = retriever
      self.ranker = ranker
      self.reranker = reranker

  def recommend(self, user):

      request = RetrievalRequest(
          user_id=user.user_id,
          k=100
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

      return candidates

