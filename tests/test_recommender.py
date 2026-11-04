"""Small deterministic checks for the recommender's data and cache contracts."""
import unittest

import numpy as np
import pandas as pd

from recommender import (
    build_demo_data,
    make_rating_matrix,
    score_metrics,
    split_by_user_time,
)


class SplitByUserTimeTest(unittest.TestCase):
    def test_holds_out_each_user_latest_rating(self):
        ratings = pd.DataFrame({
            "userId": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "movieId": [10, 20, 30, 40, 50, 10, 20, 30, 40, 50],
            "rating": [5, 4, 3, 2, 1, 4, 5, 3, 2, 1],
            "timestamp": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        })
        train, test = split_by_user_time(ratings, 0.2)
        self.assertEqual(len(train), 8)
        self.assertEqual(len(test), 2)
        for user_id in (1, 2):
            latest = ratings.loc[ratings["userId"] == user_id, "timestamp"].max()
            held_out = test.loc[test["userId"] == user_id]
            self.assertEqual(len(held_out), 1)
            self.assertEqual(int(held_out["timestamp"].iloc[0]), int(latest))

    def test_keeps_one_training_row_for_tiny_history(self):
        ratings = pd.DataFrame({
            "userId": [1, 1],
            "movieId": [10, 20],
            "rating": [5, 4],
            "timestamp": [1, 2],
        })
        train, test = split_by_user_time(ratings, 0.2)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(test), 1)


class RatingMatrixTest(unittest.TestCase):
    def test_uses_contiguous_positions_and_nan_for_missing(self):
        ratings = pd.DataFrame({
            "userId": [1, 2],
            "movieId": [10, 30],
            "rating": [4.0, 2.5],
        })
        matrix = make_rating_matrix(ratings, [1, 2], [10, 20, 30])
        self.assertEqual(matrix.shape, (2, 3))
        self.assertEqual(matrix[0, 0], 4.0)
        self.assertEqual(matrix[1, 2], 2.5)
        self.assertTrue(np.isnan(matrix[0, 1]))


class MetricsTest(unittest.TestCase):
    def test_perfect_prediction(self):
        metrics = score_metrics([1.0, 2.0], [1.0, 2.0])
        self.assertEqual(metrics["RMSE"], 0.0)
        self.assertEqual(metrics["MAE"], 0.0)

    def test_known_errors(self):
        metrics = score_metrics([1.0, 3.0], [2.0, 1.0])
        self.assertAlmostEqual(metrics["RMSE"], np.sqrt(2.5))
        self.assertAlmostEqual(metrics["MAE"], 1.5)


class CacheBuildTest(unittest.TestCase):
    def test_builds_unseen_recommendations(self):
        ratings = pd.DataFrame({
            "userId": [1, 1, 2, 2, 2, 3, 3, 3],
            "movieId": [10, 20, 10, 20, 30, 10, 30, 40],
            "rating": [5, 4, 4, 5, 4, 5, 4, 5],
            "timestamp": [1, 2, 1, 2, 3, 1, 2, 3],
        })
        movies = pd.DataFrame({
            "movieId": [10, 20, 30, 40],
            "title": ["A", "B", "C", "D"],
            "genres": ["Drama"] * 4,
        })
        params = {
            "bias": {"lambda_user": 0.0, "lambda_movie": 0.0},
            "user_cf": {
                "k_neighbors": 2,
                "min_common": 1,
                "shrinkage": 0.0,
                "correction_scale": 1.0,
            },
            "item_cf": {
                "k_neighbors": 2,
                "min_common": 1,
                "shrinkage": 0.0,
                "correction_scale": 1.0,
            },
            "hybrid_scale": 1.0,
        }
        cache = build_demo_data(ratings, movies, params=params, metrics={}, recommendation_limit=2)
        self.assertEqual(len(cache["users"]), 3)
        self.assertTrue(any(user["recommendations"] for user in cache["users"].values()))
        for user_id, user in cache["users"].items():
            seen = set(ratings.loc[ratings["userId"] == int(user_id), "movieId"])
            recommended = {movie["movieId"] for movie in user["recommendations"]}
            self.assertLessEqual(len(recommended), 2)
            self.assertFalse(recommended & seen)


if __name__ == "__main__":
    unittest.main()
