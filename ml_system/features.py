class RatingFeatureComputer:

    def __init__(self):
        self.rating_sum = {}
        self.rating_count = {}

    def update(self, entity_id, rating):

        self.rating_sum[entity_id] = (
            self.rating_sum.get(entity_id, 0) + rating
        )

        self.rating_count[entity_id] = (
            self.rating_count.get(entity_id, 0) + 1
        )

        return (
            self.rating_sum[entity_id] /
            self.rating_count[entity_id]
        )

class ItemRatingFeatureComputer:

    def __init__(self):
        self.rating_sum = {}
        self.rating_count = {}

    def update(self, movie_id, rating):

        self.rating_sum[movie_id] = (
            self.rating_sum.get(movie_id, 0) + rating
        )

        self.rating_count[movie_id] = (
            self.rating_count.get(movie_id, 0) + 1
        )

        return (
            self.rating_sum[movie_id] /
            self.rating_count[movie_id]
        )