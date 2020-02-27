"""Evaluate Implicit Recommendation models."""
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import average_precision_at_k, dcg_at_k, recall_at_k


class PredictRankings:
    """Predict rankings by trained recommendations."""

    def __init__(self, user_embed: np.ndarray, item_embed: np.ndarray) -> None:
        """Initialize Class."""
        # latent embeddings
        self.user_embed = user_embed
        self.item_embed = item_embed

    def predict(self, users: np.array, items: np.array) -> np.ndarray:
        """Predict scores for each user-item pairs."""
        # predict ranking score for each user
        user_emb = self.user_embed[users].reshape(1, self.user_embed.shape[1])
        item_emb = self.item_embed[items]
        scores = (user_emb @ item_emb.T).flatten()
        return scores


metrics = {'DCG': dcg_at_k,
           'Recall': recall_at_k,
           'MAP': average_precision_at_k}


def aoa_evaluator(user_embed: np.ndarray,
                  item_embed: np.ndarray,
                  test: np.ndarray,
                  model_name: str,
                  at_k: List[int] = [1, 3, 5]) -> pd.DataFrame:
    """Calculate ranking metrics with average-over-all evaluator."""
    # test data
    users = test[:, 0]
    items = test[:, 1]
    relevances = 0.001 + 0.999 * test[:, 2]

    # define model
    model = PredictRankings(user_embed=user_embed, item_embed=item_embed)

    # prepare ranking metrics
    results = {}
    for k in at_k:
        for metric in metrics:
            results[f'{metric}@{k}'] = []

    # calculate ranking metrics
    np.random.seed(12345)
    for user in set(users):
        indices = users == user
        pos_items = items[indices]
        rel = relevances[indices]

        # predict ranking score for each user
        scores = model.predict(users=user, items=pos_items)
        for k in at_k:
            for metric, metric_func in metrics.items():
                results[f'{metric}@{k}'].append(metric_func(rel, scores, k))

        # aggregate results
        results_df = pd.DataFrame(index=results.keys())
        results_df[model_name] = \
            list(map(np.mean, list(results.values())))

    return results_df


def unbiased_evaluator(user_embed: np.ndarray,
                       item_embed: np.ndarray,
                       train: np.ndarray,
                       val: np.ndarray,
                       pscore: np.ndarray,
                       metric: str = 'DCG',
                       num_negatives: int = 100,
                       k: int = 5) -> float:
    """Calculate ranking metrics by unbiased evaluator."""
    # validation data
    users = val[:, 0]
    items = val[:, 1]
    positive_pairs = np.r_[train[train[:, 2] == 1, :2], val]

    # define model
    model = PredictRankings(user_embed=user_embed, item_embed=item_embed)

    # calculate an unbiased DCG score
    dcg_values = list()
    unique_items = np.unique(items)
    np.random.seed(12345)
    for user in set(users):
        indices = users == user
        pos_items = items[indices]
        all_pos_items = positive_pairs[positive_pairs[:, 0] == user]
        neg_items = np.random.permutation(
            np.setdiff1d(unique_items, all_pos_items))[:num_negatives]
        used_items = np.r_[pos_items, neg_items]
        pscore_ = pscore[used_items]
        relevances = np.r_[np.ones_like(pos_items), np.zeros(num_negatives)]

        # calculate an unbiased DCG score for a user
        scores = model.predict(users=user, items=used_items)
        dcg_values.append(metrics[metric](relevances, scores, k, pscore_))

    return np.mean(dcg_values)
