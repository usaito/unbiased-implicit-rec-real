"""Evaluate Implicit Recommendation models."""
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats

from metrics import average_precision_at_k, dcg_at_k, recall_at_k


class Model:
    """Load learned recommendation models."""

    def __init__(self, model_name: str):
        """Initialize Class."""
        self.model_name = model_name

        # load learnt embeddings
        self.user_embed = np.load(
            f'../logs/{self.model_name}/embeds/user_embed.npy')
        self.item_embed = np.load(
            f'../logs/{self.model_name}/embeds/item_embed.npy')

    def predict(self, users: np.array, items: np.array) -> np.array:
        """Predict scores for each user-item pairs."""
        # predict ranking score for each user
        user_emb = self.user_embed[users].reshape(1, self.user_embed.shape[1])
        item_emb = self.item_embed[items]
        scores = (user_emb @ item_emb.T).flatten()
        return scores


def evaluate(train: np.ndarray, val: np.ndarray, test: np.ndarray,
             model_name: str, rare: int = 500, k: List[int] = [1, 3, 5]) -> None:
    """Evaluate a Recommender."""
    # positive data
    pos_data = np.r_[train[train[:, 2] == 1, :2], val, test[:, :2]]
    # estimate pscore
    matrix = sparse.lil_matrix((pos_data[:, 0].max() + 1,
                                pos_data[:, 1].max() + 1))
    for (u, i) in pos_data:
        matrix[u, i] = 1
    item_freq = np.array(matrix.sum(axis=0)).flatten()
    test_rare = test[item_freq[test[:, 1]] <= rare]
    for test, _ in zip([test_rare, test], ['rare', 'all']):
        evaluator = AverageOverAllEvaluator(
            test=test, pos_data=pos_data, model_name=model_name)
        evaluator.evaluate(k=k, rare=_)


class AverageOverAllEvaluator:
    """Average-Over-All Evaluator."""

    def __init__(self, test: np.array, pos_data: np.array,
                 model_name: str, save: bool = True) -> None:
        """Initialize class."""
        self.model = Model(model_name)
        self.users = test[:, 0]
        self.items = test[:, 1]
        self.ratings = 0.01 + 0.99 * test[:, 2]
        self.save = save
        self.unique_items = np.unique(self.items)
        self.pos_data = pos_data

    def evaluate(self, k: List[int] = [1, 3, 5], rare: str = 'all') -> None:
        """Evaluate a Recommender with the naive estimator."""
        results = {}
        metrics = {'DCG': dcg_at_k,
                   'Recall': recall_at_k,
                   'MAP': average_precision_at_k}

        for _k in k:
            for metric in metrics:
                results[f'{metric}@{_k}'] = []

        np.random.seed(12345)
        for user in set(self.users):
            # create tested item-rating for each user
            indices = self.users == user
            pos_items = self.items[indices]
            ratings = self.ratings[indices]

            # predict ranking score for each user
            scores = self.model.predict(users=user, items=pos_items)
            for _k in k:
                for metric, metric_func in metrics.items():
                    results[f'{metric}@{_k}'].append(
                        metric_func(ratings, scores, _k))

            # aggregate results
            self.results = pd.DataFrame(index=results.keys())
            self.results[f'{self.model.model_name}'] = list(
                map(np.mean, list(results.values())))
            # save results
            ret_path = Path(f'../logs/{self.model.model_name}/results/')
            ret_path.mkdir(parents=True, exist_ok=True)
            if self.save:
                self.results.to_csv(str(ret_path / f'aoa_{rare}.csv'))


class UnbiasedEvaluator:
    """Unbiased Evaluator."""

    def __init__(
            self, train: np.array, val: np.array, pscore: np.array,
            model_name: str, save: bool = True) -> None:
        """Initialize class."""
        self.model = Model(model_name)
        self.users = val[:, 0]
        self.items = val[:, 1]
        self.pscore = pscore
        self.pos_data = np.r_[train[train[:, 2] == 1, :2], val]

    def evaluate(self, num_negatives: int = 100, k: int = 5) -> None:
        """Evaluate a Recommender with the unbiased estimator."""
        results = {}
        metrics = {'DCG': dcg_at_k}

        for metric in metrics:
            results[f'{metric}@{k}'] = []

        unique_item_ids = np.unique(self.items)
        np.random.seed(12345)
        for user in set(self.users):
            # create tested item-rating for each user
            indices = self.users == user
            items_for_current_user = self.items[indices]
            all_pos_items_for_current_user = \
                self.pos_data[self.pos_data[:, 0] == user]
            neg_item = np.random.permutation(
                np.setdiff1d(unique_item_ids, all_pos_items_for_current_user))[:num_negatives]
            items_for_current_user = np.r_[items_for_current_user, neg_item]
            pscore_for_current_user = self.pscore[items_for_current_user]
            ratings_for_current_user = \
                np.r_[np.ones_like(items_for_current_user),
                      np.zeros(num_negatives)]

            # predict ranking score for each user
            scores = self.model.predict(
                users=user, items=items_for_current_user)
            for metric, metric_func in metrics.items():
                results[f'{metric}@{k}'].append(
                    metric_func(ratings_for_current_user, scores,
                                k, pscore_for_current_user))

            # aggregate results
            self.results = pd.DataFrame(index=results.keys())
            self.results[f'{self.model.model_name}'] = list(
                map(np.mean, list(results.values())))
