from dataclasses import dataclass

@dataclass
class User:
    user_id: int
    age: int
    gender: str
    occupation: str
    zip_code: str


@dataclass
class Movie:
    movie_id: int
    title: str
    release_date: str
    genres: list[str]


@dataclass
class Rating:
    user_id: int
    movie_id: int
    rating: int
    timestamp: int

class UserState: #bottom

    def __init__(self, user):

        self.user_id = user.user_id

        # Ground truth
        self.ratings = {}
        self.watched_movies = set()

        # Derived from history
        self.genre_preferences = {}
        self.average_rating = 0.0
        self.activity_level = 0.0

        # Simulation
        self.session_count = 0
        self.last_active = None

class MovieState:

  def __init__(self, movie):

      self.movie_id = movie.movie_id

      # Static
      self.title = movie.title
      self.genres = movie.genres

      # Dynamic
      self.impressions = 0
      self.clicks = 0
      self.watches = 0
      self.ratings = []

      self.num_ratings = 0
      self.average_rating = 0.0
      self.popularity = 0
      self.trending_score = 0