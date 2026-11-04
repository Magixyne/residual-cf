"""MovieLens collaborative filtering, controlled evaluation, and cached recommendations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def load_data(data_dir: Path):
    data_dir = Path(data_dir)
    return (
        pd.read_csv(data_dir / "ratings.csv"),
        pd.read_csv(data_dir / "movies.csv"),
    )


def split_by_user_time(data: pd.DataFrame, test_fraction: float = 0.2):
    """Hold out each user's latest ratings while keeping at least one training row."""
    train_idx, heldout_idx = [], []
    for _, group in data.groupby("userId", sort=True):
        ordered = group.sort_values(["timestamp", "movieId"], kind="stable")
        indices = ordered.index.to_numpy()
        heldout_count = min(
            len(indices) - 1,
            max(1, int(np.ceil(len(indices) * test_fraction))),
        )
        train_idx.extend(indices[:-heldout_count])
        heldout_idx.extend(indices[-heldout_count:])
    return (
        data.loc[train_idx].reset_index(drop=True),
        data.loc[heldout_idx].reset_index(drop=True),
    )


def make_rating_matrix(data, user_ids, movie_ids):
    """Use contiguous positions; NaN is missing and never treated as a zero rating."""
    user_pos = pd.Index(user_ids).get_indexer(data["userId"])
    movie_pos = pd.Index(movie_ids).get_indexer(data["movieId"])
    if (user_pos < 0).any() or (movie_pos < 0).any():
        raise ValueError("Input contains a userId or movieId outside the index maps.")
    matrix = np.full((len(user_ids), len(movie_ids)), np.nan, dtype=np.float32)
    matrix[user_pos, movie_pos] = data["rating"].to_numpy(dtype=np.float32)
    return matrix


def _estimate_biases(rows, cols, values, n_users, n_movies, lambda_user, lambda_movie, epochs=10):
    """Alternating ridge-style user and movie bias estimates."""
    global_mean = float(values.mean())
    user_count = np.bincount(rows, minlength=n_users).astype(np.float32)
    movie_count = np.bincount(cols, minlength=n_movies).astype(np.float32)
    user_bias = np.zeros(n_users, dtype=np.float32)
    movie_bias = np.zeros(n_movies, dtype=np.float32)
    for _ in range(epochs):
        user_numerator = np.bincount(
            rows,
            weights=values - global_mean - movie_bias[cols],
            minlength=n_users,
        )
        movie_numerator = np.bincount(
            cols,
            weights=values - global_mean - user_bias[rows],
            minlength=n_movies,
        )
        user_denominator = lambda_user + user_count
        movie_denominator = lambda_movie + movie_count
        user_bias = np.divide(
            user_numerator, user_denominator,
            out=np.zeros(n_users, dtype=np.float32), where=user_denominator > 0,
        )
        movie_bias = np.divide(
            movie_numerator, movie_denominator,
            out=np.zeros(n_movies, dtype=np.float32), where=movie_denominator > 0,
        )
    return global_mean, user_bias, movie_bias


def fit_bias_and_similarity(
    train_data,
    user_ids,
    movie_ids,
    lambda_user=10.0,
    lambda_movie=25.0,
    epochs=10,
):
    """Fit regularized biases, residuals, and user-to-user similarities."""
    matrix = make_rating_matrix(train_data, user_ids, movie_ids)
    observed = np.isfinite(matrix)
    rows, cols = np.nonzero(observed)
    values = matrix[rows, cols]
    global_mean, user_bias, movie_bias = _estimate_biases(
        rows, cols, values, len(user_ids), len(movie_ids),
        lambda_user, lambda_movie, epochs,
    )
    baseline = global_mean + user_bias[:, None] + movie_bias[None, :]
    residual = np.zeros_like(matrix, dtype=np.float32)
    residual[rows, cols] = values - baseline[rows, cols]
    norms = np.sqrt(np.sum(residual * residual, axis=1, dtype=np.float64)).astype(np.float32)
    dot = residual @ residual.T
    norm_product = norms[:, None] * norms[None, :]
    similarity = np.divide(dot, norm_product, out=np.zeros_like(dot), where=norm_product > 0)
    overlap = observed.astype(np.int32) @ observed.astype(np.int32).T
    np.fill_diagonal(similarity, 0.0)
    return {
        "global_mean": global_mean,
        "user_bias": user_bias,
        "movie_bias": movie_bias,
        "baseline": baseline,
        "residual": residual,
        "observed": observed,
        "similarity": similarity,
        "overlap": overlap,
        "rating_min": float(train_data["rating"].min()),
        "rating_max": float(train_data["rating"].max()),
    }


def prepare_user_similarity(fit, k_neighbors, min_common, shrinkage):
    similarity = fit["similarity"].copy()
    overlap = fit["overlap"]
    similarity[overlap < min_common] = 0.0
    if shrinkage > 0:
        similarity *= overlap / (overlap + shrinkage)
    np.maximum(similarity, 0.0, out=similarity)
    np.fill_diagonal(similarity, 0.0)
    for user_row in range(len(similarity)):
        positive = np.flatnonzero(similarity[user_row] > 0)
        if len(positive) > k_neighbors:
            keep = positive[np.argpartition(similarity[user_row, positive], -k_neighbors)[-k_neighbors:]]
            keep_mask = np.zeros(len(similarity), dtype=bool)
            keep_mask[keep] = True
            similarity[user_row, ~keep_mask] = 0.0
    return similarity


def residual_correction(fit, k_neighbors, min_common, shrinkage):
    """Compute top-K positive user-neighbor residual corrections."""
    similarity = prepare_user_similarity(fit, k_neighbors, min_common, shrinkage)
    weights = similarity @ fit["observed"].astype(np.float32)
    weighted_residual = similarity @ fit["residual"]
    correction = np.divide(
        weighted_residual,
        weights,
        out=np.zeros_like(weighted_residual),
        where=weights > 0,
    )
    return correction, weights


def validation_user_correction(
    fit,
    validation_user_pos,
    validation_movie_pos,
    validation_observed,
    validation_residual,
    k_neighbors,
    min_common,
    shrinkage,
):
    """Compute validation-pair corrections without materializing full predictions."""
    similarity = prepare_user_similarity(fit, k_neighbors, min_common, shrinkage)
    pair_weights = similarity[validation_user_pos] * validation_observed
    denominator = pair_weights.sum(axis=1)
    numerator = (pair_weights * validation_residual).sum(axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def build_item_candidates(fit, candidate_count=128, overlap_chunk=50_000):
    """Find item neighbors sparsely, then count shared raters only for those pairs."""
    residual_by_item = csr_matrix(fit["residual"].T)
    n_items = residual_by_item.shape[0]
    n_neighbors = min(n_items, candidate_count + 1)
    search = NearestNeighbors(
        metric="cosine", algorithm="brute", n_jobs=1, n_neighbors=n_neighbors,
    )
    search.fit(residual_by_item)
    distance, indices = search.kneighbors(residual_by_item, return_distance=True)
    similarities = np.maximum(0.0, 1.0 - distance).astype(np.float32)

    targets = np.repeat(np.arange(n_items, dtype=np.int32), n_neighbors)
    neighbors = indices.reshape(-1).astype(np.int32, copy=False)
    counts = np.zeros(len(neighbors), dtype=np.uint16)
    valid = targets != neighbors
    observed = fit["observed"]
    valid_positions = np.flatnonzero(valid)
    for start in range(0, len(valid_positions), overlap_chunk):
        positions = valid_positions[start:start + overlap_chunk]
        counts[positions] = (
            observed[:, targets[positions]] & observed[:, neighbors[positions]]
        ).sum(axis=0, dtype=np.uint16)
    counts = counts.reshape(n_items, n_neighbors)
    similarities[~valid.reshape(n_items, n_neighbors)] = 0.0
    return {"indices": indices, "similarities": similarities, "overlap": counts}


def item_similarity_from_candidates(candidates, k_neighbors, min_common, shrinkage):
    """Shrink and retain top-K directed item neighbors as a sparse matrix."""
    from scipy.sparse import csr_matrix

    indices = candidates["indices"]
    similarities = candidates["similarities"]
    overlap = candidates["overlap"]
    n_items = indices.shape[0]
    rows, cols, values = [], [], []
    for item in range(n_items):
        neighbors = indices[item]
        counts = overlap[item].astype(np.float32)
        weights = similarities[item] * np.divide(
            counts, counts + shrinkage, out=np.zeros_like(counts), where=counts > 0,
        )
        valid = (neighbors != item) & (counts >= min_common) & (weights > 0)
        selected_neighbors = neighbors[valid]
        selected_weights = weights[valid]
        if len(selected_weights) > k_neighbors:
            selected = np.argpartition(selected_weights, -k_neighbors)[-k_neighbors:]
            selected_neighbors = selected_neighbors[selected]
            selected_weights = selected_weights[selected]
        rows.extend([item] * len(selected_neighbors))
        cols.extend(selected_neighbors.tolist())
        values.extend(selected_weights.tolist())
    return csr_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)


def item_correction_at_pairs(fit, item_similarity, user_pos, movie_pos):
    """Predict residual corrections only for requested user-item pairs."""
    residual = csr_matrix(fit["residual"])
    observed = csr_matrix(fit["observed"].astype(np.float32))
    item_similarity_t = item_similarity.T.tocsr()
    weighted_residual = residual @ item_similarity_t
    support = observed @ item_similarity_t
    numerator = np.asarray(weighted_residual[user_pos, movie_pos]).reshape(-1)
    denominator = np.asarray(support[user_pos, movie_pos]).reshape(-1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def full_item_correction(fit, item_similarity):
    """Compute dense user-item item-CF correction and support for recommendation ranking."""
    residual = csr_matrix(fit["residual"])
    observed = csr_matrix(fit["observed"].astype(np.float32))
    item_similarity_t = item_similarity.T.tocsr()
    weighted_residual = (residual @ item_similarity_t).toarray()
    support = (observed @ item_similarity_t).toarray()
    correction = np.divide(
        weighted_residual, support, out=np.zeros_like(weighted_residual), where=support > 0,
    )
    return correction.astype(np.float32), support.astype(np.float32)


def score_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float32)
    predicted = np.asarray(predicted, dtype=np.float32)
    return {
        "RMSE": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "MAE": float(np.mean(np.abs(actual - predicted))),
    }


def tune_bias_parameters(train_data, validation, user_ids, movie_ids):
    """Tune both baselines rather than giving the proposed CF model extra attention."""
    user_index, movie_index = pd.Index(user_ids), pd.Index(movie_ids)
    rows = user_index.get_indexer(train_data["userId"])
    cols = movie_index.get_indexer(train_data["movieId"])
    values = train_data["rating"].to_numpy(dtype=np.float32)
    val_users = user_index.get_indexer(validation["userId"])
    val_movies = movie_index.get_indexer(validation["movieId"])
    actual = validation["rating"].to_numpy(dtype=np.float32)
    user_count = np.bincount(rows, minlength=len(user_ids)).astype(np.float32)
    movie_count = np.bincount(cols, minlength=len(movie_ids)).astype(np.float32)
    best = None
    for lambda_user in [0.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        for lambda_movie in [0.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
            mean, user_bias, movie_bias = _estimate_biases(
                rows, cols, values, len(user_ids), len(movie_ids),
                lambda_user, lambda_movie,
            )
            prediction = np.clip(mean + user_bias[val_users] + movie_bias[val_movies], 0.5, 5.0)
            metrics = score_metrics(actual, prediction)
            candidate = (metrics["RMSE"], metrics["MAE"], lambda_user, lambda_movie)
            if best is None or candidate < best:
                best = candidate
    return {"lambda_user": best[2], "lambda_movie": best[3]}, best[:2]


def tune_user_cf(fit, validation, user_pos, movie_pos, baseline_prediction):
    observed = fit["observed"][:, movie_pos].T
    residual = fit["residual"][:, movie_pos].T
    actual = validation["rating"].to_numpy(dtype=np.float32)
    rows = []
    for k in [20, 40, 80, 160, 320, 500]:
        for min_common in [3, 5, 10]:
            for shrinkage in [0.0, 5.0, 15.0]:
                correction = validation_user_correction(
                    fit, user_pos, movie_pos, observed, residual,
                    k, min_common, shrinkage,
                )
                for scale in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                    prediction = np.clip(
                        baseline_prediction + scale * correction,
                        fit["rating_min"], fit["rating_max"],
                    )
                    rows.append({
                        "k_neighbors": k,
                        "min_common": min_common,
                        "shrinkage": shrinkage,
                        "correction_scale": scale,
                        **score_metrics(actual, prediction),
                    })
    rows.sort(key=lambda row: (row["RMSE"], row["MAE"]))
    best = rows[0]
    params = {key: best[key] for key in ("k_neighbors", "min_common", "shrinkage", "correction_scale")}
    params["k_neighbors"] = int(params["k_neighbors"])
    params["min_common"] = int(params["min_common"])
    return params, rows[:5]


def tune_item_cf(fit, candidates, validation, user_pos, movie_pos, baseline_prediction):
    actual = validation["rating"].to_numpy(dtype=np.float32)
    rows = []
    for k in [20, 50, 100]:
        for min_common in [1, 2, 3]:
            for shrinkage in [5.0, 15.0]:
                similarity = item_similarity_from_candidates(candidates, k, min_common, shrinkage)
                correction = item_correction_at_pairs(fit, similarity, user_pos, movie_pos)
                for scale in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    prediction = np.clip(
                        baseline_prediction + scale * correction,
                        fit["rating_min"], fit["rating_max"],
                    )
                    rows.append({
                        "k_neighbors": k,
                        "min_common": min_common,
                        "shrinkage": shrinkage,
                        "correction_scale": scale,
                        **score_metrics(actual, prediction),
                    })
    rows.sort(key=lambda row: (row["RMSE"], row["MAE"]))
    best = rows[0]
    params = {key: best[key] for key in ("k_neighbors", "min_common", "shrinkage", "correction_scale")}
    params["k_neighbors"] = int(params["k_neighbors"])
    params["min_common"] = int(params["min_common"])
    return params, rows[:5]


def synthetic_positive_control():
    """Check that a planted similar-user signal moves a withheld prediction upward."""
    residual = np.array([
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 2.0],
        [-1.0, -1.0, -2.0],
    ], dtype=np.float32)
    observed = np.array([
        [True, True, False],
        [True, True, True],
        [True, True, True],
    ])
    norms = np.sqrt(np.sum(residual * residual, axis=1))
    similarity = residual @ residual.T / np.maximum(norms[:, None] * norms[None, :], 1e-12)
    overlap = observed.astype(np.int32) @ observed.astype(np.int32).T
    np.fill_diagonal(similarity, 0.0)
    fit = {
        "similarity": similarity,
        "overlap": overlap,
        "observed": observed,
        "residual": residual,
        "rating_min": 0.5,
        "rating_max": 5.0,
        "baseline": np.full((3, 3), 3.0, dtype=np.float32),
    }
    correction, _ = residual_correction(fit, 2, 2, 0.0)
    prediction = float(np.clip(fit["baseline"][0, 2] + correction[0, 2], 0.5, 5.0))
    baseline = float(fit["baseline"][0, 2])
    return {
        "passed": bool(prediction > baseline + 1.0),
        "baseline": baseline,
        "prediction": prediction,
        "gain": prediction - baseline,
    }


def randomize_similarity_weights(similarity, rng):
    """Preserve each user's weight distribution and degree, randomize neighbor identity."""
    n_users = similarity.shape[0]
    null_similarity = np.zeros_like(similarity)
    all_users = np.arange(n_users)
    for user in range(n_users):
        weights = similarity[user, similarity[user] > 0]
        if len(weights) == 0:
            continue
        pool = all_users[all_users != user]
        destinations = rng.choice(pool, size=min(len(weights), len(pool)), replace=False)
        null_similarity[user, destinations] = rng.permutation(weights[:len(destinations)])
    return null_similarity


def negative_control_metrics(fit, test, user_pos, movie_pos, base_prediction, user_params, repeats=25):
    """Random-neighbor placebo: preserve score weights but break user-neighbor links."""
    real_similarity = prepare_user_similarity(
        fit,
        user_params["k_neighbors"],
        user_params["min_common"],
        user_params["shrinkage"],
    )
    observed = fit["observed"][:, movie_pos].T.astype(np.float32)
    residual = fit["residual"][:, movie_pos].T
    actual = test["rating"].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(80421)
    metrics = []
    for _ in range(repeats):
        null_similarity = randomize_similarity_weights(real_similarity, rng)
        pair_weights = null_similarity[user_pos] * observed
        denominator = pair_weights.sum(axis=1)
        numerator = (pair_weights * residual).sum(axis=1)
        correction = np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0,
        )
        prediction = np.clip(
            base_prediction + user_params["correction_scale"] * correction,
            fit["rating_min"], fit["rating_max"],
        )
        metrics.append(score_metrics(actual, prediction))
    rmses = np.array([row["RMSE"] for row in metrics])
    maes = np.array([row["MAE"] for row in metrics])
    return {
        "repeats": repeats,
        "RMSE_mean": float(rmses.mean()),
        "RMSE_p05": float(np.quantile(rmses, 0.05)),
        "RMSE_p95": float(np.quantile(rmses, 0.95)),
        "MAE_mean": float(maes.mean()),
        "MAE_p05": float(np.quantile(maes, 0.05)),
        "MAE_p95": float(np.quantile(maes, 0.95)),
    }


def paired_user_bootstrap(actual, baseline_prediction, model_prediction, user_ids, draws=2000):
    """Cluster bootstrap users; paired errors use exactly the same held-out rows."""
    unique_users, row_user = np.unique(user_ids, return_inverse=True)
    n_users = len(unique_users)
    counts = np.bincount(row_user, minlength=n_users).astype(np.float64)
    baseline_error = np.asarray(actual) - np.asarray(baseline_prediction)
    model_error = np.asarray(actual) - np.asarray(model_prediction)
    base_sse = np.bincount(row_user, weights=baseline_error ** 2, minlength=n_users)
    model_sse = np.bincount(row_user, weights=model_error ** 2, minlength=n_users)
    base_sae = np.bincount(row_user, weights=np.abs(baseline_error), minlength=n_users)
    model_sae = np.bincount(row_user, weights=np.abs(model_error), minlength=n_users)
    rng = np.random.default_rng(20261104)
    sampled = rng.integers(0, n_users, size=(draws, n_users))
    n = counts[sampled].sum(axis=1)
    rmse_gain = np.sqrt(base_sse[sampled].sum(axis=1) / n) - np.sqrt(model_sse[sampled].sum(axis=1) / n)
    mae_gain = base_sae[sampled].sum(axis=1) / n - model_sae[sampled].sum(axis=1) / n
    return {
        "users": int(n_users),
        "draws": int(draws),
        "delta_RMSE_95CI": [float(x) for x in np.quantile(rmse_gain, [0.025, 0.975])],
        "delta_MAE_95CI": [float(x) for x in np.quantile(mae_gain, [0.025, 0.975])],
    }


def evaluate_model(ratings):
    """Use nested time splits; tune every baseline/model before a paired outer test."""
    user_ids = np.sort(ratings["userId"].unique())
    movie_ids = np.sort(ratings["movieId"].unique())
    user_index, movie_index = pd.Index(user_ids), pd.Index(movie_ids)
    outer_train, test = split_by_user_time(ratings, 0.2)
    inner_train, validation = split_by_user_time(outer_train, 0.2)
    val_user_pos = user_index.get_indexer(validation["userId"])
    val_movie_pos = movie_index.get_indexer(validation["movieId"])
    val_actual = validation["rating"].to_numpy(dtype=np.float32)

    bias_params, bias_validation = tune_bias_parameters(
        inner_train, validation, user_ids, movie_ids,
    )
    inner_fit = fit_bias_and_similarity(
        inner_train, user_ids, movie_ids,
        bias_params["lambda_user"], bias_params["lambda_movie"],
    )
    val_baseline = inner_fit["baseline"][val_user_pos, val_movie_pos]
    user_params, user_grid = tune_user_cf(
        inner_fit, validation, val_user_pos, val_movie_pos, val_baseline,
    )
    item_candidates_inner = build_item_candidates(inner_fit)
    item_params, item_grid = tune_item_cf(
        inner_fit, item_candidates_inner, validation,
        val_user_pos, val_movie_pos, val_baseline,
    )
    user_validation_correction = validation_user_correction(
        inner_fit,
        val_user_pos,
        val_movie_pos,
        inner_fit["observed"][:, val_movie_pos].T,
        inner_fit["residual"][:, val_movie_pos].T,
        user_params["k_neighbors"],
        user_params["min_common"],
        user_params["shrinkage"],
    )
    item_similarity_inner = item_similarity_from_candidates(
        item_candidates_inner,
        item_params["k_neighbors"],
        item_params["min_common"],
        item_params["shrinkage"],
    )
    item_validation_correction = item_correction_at_pairs(
        inner_fit, item_similarity_inner, val_user_pos, val_movie_pos,
    )
    user_delta_val = user_params["correction_scale"] * user_validation_correction
    item_delta_val = item_params["correction_scale"] * item_validation_correction
    hybrid_rows = []
    for hybrid_scale in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]:
        prediction = np.clip(
            val_baseline + hybrid_scale * (user_delta_val + item_delta_val), 0.5, 5.0,
        )
        hybrid_rows.append({
            "hybrid_scale": hybrid_scale,
            **score_metrics(val_actual, prediction),
        })
    hybrid_rows.sort(key=lambda row: (row["RMSE"], row["MAE"]))
    hybrid_scale = float(hybrid_rows[0]["hybrid_scale"])
    params = {
        "bias": bias_params,
        "user_cf": user_params,
        "item_cf": item_params,
        "hybrid_scale": hybrid_scale,
    }

    outer_fit = fit_bias_and_similarity(
        outer_train, user_ids, movie_ids,
        bias_params["lambda_user"], bias_params["lambda_movie"],
    )
    user_correction, user_support = residual_correction(
        outer_fit,
        user_params["k_neighbors"],
        user_params["min_common"],
        user_params["shrinkage"],
    )
    item_candidates_outer = build_item_candidates(outer_fit)
    item_similarity_outer = item_similarity_from_candidates(
        item_candidates_outer,
        item_params["k_neighbors"],
        item_params["min_common"],
        item_params["shrinkage"],
    )
    item_correction, item_support = full_item_correction(outer_fit, item_similarity_outer)

    test_user_pos = user_index.get_indexer(test["userId"])
    test_movie_pos = movie_index.get_indexer(test["movieId"])
    actual = test["rating"].to_numpy(dtype=np.float32)
    global_prediction = np.full(len(test), outer_fit["global_mean"], dtype=np.float32)
    bias_prediction = outer_fit["baseline"][test_user_pos, test_movie_pos]
    user_means = outer_train.groupby("userId")["rating"].mean()
    user_mean_prediction = test["userId"].map(user_means).to_numpy(dtype=np.float32)
    user_delta = user_params["correction_scale"] * user_correction[test_user_pos, test_movie_pos]
    item_delta = item_params["correction_scale"] * item_correction[test_user_pos, test_movie_pos]
    user_prediction = np.clip(bias_prediction + user_delta, 0.5, 5.0)
    item_prediction = np.clip(bias_prediction + item_delta, 0.5, 5.0)
    hybrid_prediction = np.clip(
        bias_prediction + hybrid_scale * (user_delta + item_delta), 0.5, 5.0,
    )
    remove_user_prediction = np.clip(bias_prediction + hybrid_scale * item_delta, 0.5, 5.0)
    remove_item_prediction = np.clip(bias_prediction + hybrid_scale * user_delta, 0.5, 5.0)
    metrics = {
        "空白对照（全局均值）": score_metrics(actual, global_prediction),
        "用户均分基线": score_metrics(actual, user_mean_prediction),
        "偏差基线": score_metrics(actual, bias_prediction),
        "加法：偏差+UserCF": score_metrics(actual, user_prediction),
        "互补：偏差+ItemCF": score_metrics(actual, item_prediction),
        "加法：偏差+UserCF+ItemCF": score_metrics(actual, hybrid_prediction),
        "减法：混合模型去掉UserCF": score_metrics(actual, remove_user_prediction),
        "减法：混合模型去掉ItemCF": score_metrics(actual, remove_item_prediction),
        "减法：混合模型去掉两项": score_metrics(actual, bias_prediction),
    }
    negative = negative_control_metrics(
        outer_fit, test, test_user_pos, test_movie_pos, bias_prediction, user_params,
    )
    metrics["阴性对照（随机邻居）"] = {
        "RMSE": negative["RMSE_mean"], "MAE": negative["MAE_mean"],
    }
    confidence = {
        "UserCF_vs_bias": paired_user_bootstrap(
            actual, bias_prediction, user_prediction, test["userId"].to_numpy(),
        ),
        "ItemCF_vs_bias": paired_user_bootstrap(
            actual, bias_prediction, item_prediction, test["userId"].to_numpy(),
        ),
        "Hybrid_vs_bias": paired_user_bootstrap(
            actual, bias_prediction, hybrid_prediction, test["userId"].to_numpy(),
        ),
    }
    return {
        "params": params,
        "metrics": metrics,
        "negative_control": negative,
        "positive_control": synthetic_positive_control(),
        "paired_bootstrap": confidence,
        "train_count": int(len(outer_train)),
        "test_count": int(len(test)),
        "validation_count": int(len(validation)),
        "tuning": {
            "bias_validation_RMSE_MAE": list(bias_validation),
            "user_cf_best5": user_grid,
            "item_cf_best5": item_grid,
            "hybrid_best5": hybrid_rows[:5],
        },
    }


def build_demo_data(
    ratings,
    movies,
    params,
    metrics,
    recommendation_limit=50,
):
    """Fit tuned components on all included ratings and build an offline cache."""
    user_ids = np.sort(ratings["userId"].unique())
    movie_ids = np.sort(ratings["movieId"].unique())
    user_index = pd.Index(user_ids)
    movie_index = pd.Index(movie_ids)
    bias = params["bias"]
    user_params = params["user_cf"]
    item_params = params["item_cf"]
    fit = fit_bias_and_similarity(
        ratings, user_ids, movie_ids,
        bias["lambda_user"], bias["lambda_movie"],
    )
    user_correction, user_support = residual_correction(
        fit,
        user_params["k_neighbors"],
        user_params["min_common"],
        user_params["shrinkage"],
    )
    item_candidates = build_item_candidates(fit)
    item_similarity = item_similarity_from_candidates(
        item_candidates,
        item_params["k_neighbors"],
        item_params["min_common"],
        item_params["shrinkage"],
    )
    item_correction, item_support = full_item_correction(fit, item_similarity)
    prediction = np.clip(
        fit["baseline"]
        + params["hybrid_scale"] * (
            user_params["correction_scale"] * user_correction
            + item_params["correction_scale"] * item_correction
        ),
        fit["rating_min"], fit["rating_max"],
    )
    support = user_support + item_support
    movie_info = movies.set_index("movieId")
    users = {}
    for user_id in user_ids:
        user_id = int(user_id)
        user_row = int(user_index.get_indexer([user_id])[0])
        history = ratings.loc[ratings.userId == user_id]
        seen_movie_ids = set(history["movieId"].tolist())
        seen_pos = movie_index.get_indexer(list(seen_movie_ids))
        score = prediction[user_row].copy()
        score[support[user_row] <= 0] = -np.inf
        score[seen_pos[seen_pos >= 0]] = -np.inf
        candidate_pos = np.flatnonzero(np.isfinite(score))
        order = candidate_pos[np.argsort(score[candidate_pos])[::-1][:recommendation_limit]]
        recs = []
        for pos in order:
            movie_id = int(movie_ids[pos])
            info = movie_info.loc[movie_id]
            recs.append({
                "movieId": movie_id,
                "title": str(info["title"]),
                "genres": str(info["genres"]),
                "predicted_rating": round(float(score[pos]), 3),
                "neighbor_weight": round(float(support[user_row, pos]), 3),
            })
        liked = history.sort_values(["rating", "timestamp"], ascending=[False, False]).head(3)
        top_liked = []
        for row in liked.itertuples():
            title = movie_info.loc[int(row.movieId), "title"]
            top_liked.append({"title": str(title), "rating": round(float(row.rating), 1)})
        users[str(user_id)] = {
            "rated_count": int(len(history)),
            "mean_rating": round(float(history.rating.mean()), 3),
            "top_liked": top_liked,
            "recommendations": recs,
        }
    return {
        "dataset": {
            "ratings": int(len(ratings)),
            "users": int(ratings.userId.nunique()),
            "rated_movies": int(ratings.movieId.nunique()),
            "catalog_movies": int(len(movies)),
        },
        "params": params,
        "recommendation_limit": int(recommendation_limit),
        "metrics": metrics or {},
        "users": users,
    }


def save_demo_data(data_dir: Path, demo_data):
    path = Path(data_dir) / "demo_data.json"
    path.write_text(json.dumps(demo_data, ensure_ascii=False), encoding="utf-8")
    return path
