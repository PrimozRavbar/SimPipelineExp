"""Per-movie behaviour signals and co-watching, from the game's own logs.

One definition, updated event by event: Flink keeps an instance online
(Flink.track_item_signals, published to Redis as ("item_signals", "all"))
and training replays the event log through another instance, so each
training row sees the signals as they were just before it (point in
time). Uses no World ground truth.

- Statistics per movie: impressions (times shown), clicks, watches,
  average completion, average rating. Smoothed towards global averages so
  rarely shown movies are not extreme. In a rich world these reveal movie
  quality, which genres don't.
- Clicks over expected clicks (COEC): a movie shown at the top gets more
  clicks whatever its quality, so its raw click rate is biased by where it
  was shown. Each impression adds the current average click rate of its
  slate position (estimated from the logs, no World knowledge) to the
  movie's expected clicks; COEC = clicks / expected clicks, smoothed
  towards 1. Above 1: clicked more than an average movie in those slots.
- Co-watching: when a user watches movie m, every movie in their last
  COWATCH_MEMORY watches gets a co-watch count with m (both ways).
  similarity(a, b) = count(a, b) / sqrt(watches(a) * watches(b)). Reveals
  hidden tastes ("people who watched X also watched Y").
"""

import math

COWATCH_MEMORY = 20
PRIOR_IMPRESSIONS = 20.0     # smoothing strength for click rate
PRIOR_WATCHES = 5.0          # ... for completion
PRIOR_RATINGS = 5.0          # ... for rating


class ItemSignals:

    def __init__(self):
        self.impressions = {}
        self.clicks = {}
        self.watches = {}
        self.completion_sum = {}
        self.rating_sum = {}
        self.rating_count = {}
        self.cowatch = {}            # movie -> {movie: count}
        self.user_watches = {}       # user -> last COWATCH_MEMORY watched movies
        self.position_impressions = {}   # slate position -> impressions
        self.position_clicks = {}        # slate position -> clicks
        self.expected_clicks = {}        # movie -> sum of position click rates at its impressions
        self.open_slates = {}            # recommendation id -> movie ids (recent slates only)
        self.totals = {"impressions": 0, "clicks": 0, "watches": 0, "completion": 0.0,
                       "ratings": 0, "rating_sum": 0.0}

    # -- updates ---------------------------------------------------------

    def observe(self, event):
        name = type(event).__name__
        if name == "RecommendationShownEvent":
            for position, m in enumerate(event.movie_ids):
                self.impressions[m] = self.impressions.get(m, 0) + 1
                self.expected_clicks[m] = self.expected_clicks.get(m, 0.0) + self.position_ctr(position)
                self.position_impressions[position] = self.position_impressions.get(position, 0) + 1
            self.totals["impressions"] += len(event.movie_ids)
            self.open_slates[event.recommendation_id] = list(event.movie_ids)
            if len(self.open_slates) > 5000:          # keep recent slates only
                for key in list(self.open_slates)[:1000]:
                    del self.open_slates[key]
        elif name == "RecommendationClickedEvent":
            self.clicks[event.movie_id] = self.clicks.get(event.movie_id, 0) + 1
            self.totals["clicks"] += 1
            slate = self.open_slates.get(event.recommendation_id)
            if slate and event.movie_id in slate:
                position = slate.index(event.movie_id)
                self.position_clicks[position] = self.position_clicks.get(position, 0) + 1
        elif name == "WatchEvent":
            m = event.movie_id
            self.watches[m] = self.watches.get(m, 0) + 1
            self.completion_sum[m] = self.completion_sum.get(m, 0.0) + event.completion
            self.totals["watches"] += 1
            self.totals["completion"] += event.completion
            history = self.user_watches.setdefault(event.user_id, [])
            for other in history:
                if other != m:
                    self.cowatch.setdefault(m, {})[other] = self.cowatch.get(m, {}).get(other, 0) + 1
                    self.cowatch.setdefault(other, {})[m] = self.cowatch.get(other, {}).get(m, 0) + 1
            history.append(m)
            del history[:-COWATCH_MEMORY]
        elif name == "RatingEvent":
            m = event.movie_id
            self.rating_sum[m] = self.rating_sum.get(m, 0.0) + event.rating
            self.rating_count[m] = self.rating_count.get(m, 0) + 1
            self.totals["ratings"] += 1
            self.totals["rating_sum"] += event.rating

    # -- reads -----------------------------------------------------------

    def position_ctr(self, position):
        """Average click rate at a slate position so far (prior 0.05)."""
        shown = self.position_impressions.get(position, 0)
        return (self.position_clicks.get(position, 0) + 0.05 * 10) / (shown + 10)

    def coec(self, movie_id, prior=5.0):
        """Clicks over expected clicks, smoothed towards 1."""
        return (self.clicks.get(movie_id, 0) + prior) / (self.expected_clicks.get(movie_id, 0.0) + prior)

    def stats(self, movie_id):
        """[smoothed click rate, smoothed completion, smoothed rating / 5,
        log(1 + impressions), clicks over expected clicks]."""
        t = self.totals
        global_ctr = t["clicks"] / t["impressions"] if t["impressions"] else 0.0
        global_completion = t["completion"] / t["watches"] if t["watches"] else 0.5
        global_rating = t["rating_sum"] / t["ratings"] if t["ratings"] else 3.0
        shown = self.impressions.get(movie_id, 0)
        ctr = (self.clicks.get(movie_id, 0) + PRIOR_IMPRESSIONS * global_ctr) / (shown + PRIOR_IMPRESSIONS)
        watched = self.watches.get(movie_id, 0)
        completion = (self.completion_sum.get(movie_id, 0.0) + PRIOR_WATCHES * global_completion) / (watched + PRIOR_WATCHES)
        rated = self.rating_count.get(movie_id, 0)
        rating = (self.rating_sum.get(movie_id, 0.0) + PRIOR_RATINGS * global_rating) / (rated + PRIOR_RATINGS)
        return [ctr, completion, rating / 5.0, math.log1p(shown), self.coec(movie_id)]

    def similarity(self, a, b):
        count = self.cowatch.get(a, {}).get(b, 0)
        if not count:
            return 0.0
        return count / math.sqrt(self.watches.get(a, 1) * self.watches.get(b, 1))

    def cowatch_score(self, movie_id, recent_watches):
        """Sum of co-watch similarities between a movie and recent watches."""
        return sum(self.similarity(w, movie_id) for w in recent_watches if w != movie_id)

    def neighbours(self, recent_watches, k):
        """Movies most co-watched with the recent watches (not themselves)."""
        scores = {}
        for w in recent_watches:
            for other in self.cowatch.get(w, {}):
                if other not in recent_watches:
                    scores[other] = scores.get(other, 0.0) + self.similarity(w, other)
        return [m for m, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]
