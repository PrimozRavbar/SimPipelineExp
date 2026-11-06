from dataclasses import dataclass

@dataclass
class Event:
    timestamp: int
    user_id: int

@dataclass
class RecommendationShownEvent(Event):
    recommendation_id: int
    movie_ids: list[int]


@dataclass
class ClickEvent(Event):
    movie_id: int

@dataclass
class RecommendationClickedEvent(Event):
    recommendation_id: int
    movie_id: int


@dataclass
class WatchEvent(Event):
    movie_id: int
    watch_time: float      # seconds
    completion: float      # 0-1


@dataclass
class RatingEvent(Event):
    movie_id: int
    rating: float

@dataclass
class Session:
    session_id: int
    user_id: int
    started_at: int

@dataclass
class RetrievalRequest:
    user_id: int
    k: int