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
      # Extra candidate sources [(source, k)] merged with the retriever's
      # candidates (see ml_system.candidate_sources). Empty = off.
      self.extra_sources = []
      # Feature logging (off by default): score_logger(tick, user,
      # slate, retriever) is called with every served slate and the
      # retriever that produced it, so features can be logged as they were
      # at serving time (and later joined to the training data).
      self.score_logger = None
      # admission() -> bool, set by KServe: False when no ready replica can
      # take the request this tick; it is then served by the fallback
      # (popularity). None = no capacity limit (the original).
      self.admission = None
      self.overflow_fallback = None
      self.sim = None
      # Movies already shown to a user earlier in the same tick (a session
      # with several slates: the next row shows different movies).
      self._session = {}
      self._session_tick = None
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
      elif self.admission is not None and not self.admission():
          fallback = self.fallback or self.overflow_fallback
          if fallback is not None:
              self.fallback_requests += 1
              retriever, ranker = fallback, None

      candidates = retriever.retrieve(request)

      if self.extra_sources and retriever is not self.fallback and retriever is not self.overflow_fallback:
          from ml_system.candidate_sources import merge_candidates
          extras = [
              source.retrieve(RetrievalRequest(user_id=user.user_id, k=k))
              for source, k in self.extra_sources
          ]
          candidates = merge_candidates(candidates, extras, self.num_candidates)

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

      now = self.sim.clock.now if self.sim is not None else None
      if now != self._session_tick:
          self._session, self._session_tick = {}, now
      seen = self._session.get(user.user_id)
      if seen:
          candidates = [m for m in candidates if m not in seen]
      slate = list(candidates)[:self.slate_size]
      if self.sim is not None and getattr(getattr(self.sim, "services", {}).get("World"), "session_slates", 1) > 1:
          self._session.setdefault(user.user_id, set()).update(slate)
      if self.score_logger is not None and self.sim is not None:
          self.score_logger(self.sim.clock.now, user, slate, retriever)
      return slate

  def start(self, sim):
      super().start(sim)
      self.sim = sim

