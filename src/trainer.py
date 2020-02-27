"""
Codes for training recommenders on semi-synthetic datasets used in the semi-synthetic experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import sparse
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops

from evaluate.evaluator import aoa_evaluator
from models.expomf import ExpoMF
from models.recommenders import PointwiseRecommender


def tocsr(data: np.array, num_user: int, num_item: int) -> sparse.csr_matrix:
    """Convert data to csr_matrix."""
    matrix = sparse.lil_matrix((num_user, num_item))
    for (u, i, r) in data[:, :3]:
        matrix[u, i] = r
    return sparse.csr_matrix(matrix)


def expomf_trainer(train: np.ndarray, num_user: int, num_item: int,
                   n_components: int = 30, lam: float = 1e-7,
                   model_name: str = 'expomf') -> Tuple[np.ndarray, np.ndarray]:
    """Train expomf models."""
    model = ExpoMF(n_components=n_components, random_state=12345,
                   save_params=False, early_stopping=True,
                   verbose=False, lam_theta=lam, lam_beta=lam)
    model.fit(tocsr(train, num_user, num_item))

    return model.theta, model.beta


def pointwise_trainer(sess: tf.Session, model: PointwiseRecommender,
                      train: np.ndarray, test: np.ndarray, pscore: np.ndarray,
                      max_iters: int = 1000, batch_size: int = 2**12,
                      model_name: str = 'rmf') -> Tuple[np.ndarray, np.ndarray]:
    """Train and Evaluate Implicit Recommender."""
    train_loss_list = []
    test_dcg_list = []
    test_map_list = []
    test_recall_list = []

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # specify model type
    ips = 'rmf' in model_name
    # pscore for train
    pscore = pscore[train[:, 1].astype(np.int)]
    # positive and unlabeled data for training set
    pos_train = train[train[:, 2] == 1]
    pscore_pos_train = pscore[train[:, 2] == 1]
    num_pos = np.sum(train[:, 2])
    unlabeled_train = train[train[:, 2] == 0]
    pscore_unlabeled_train = pscore[train[:, 2] == 0]
    num_unlabeled = np.sum(1 - train[:, 2])
    # train the given implicit recommender
    np.random.seed(12345)
    for i in np.arange(max_iters):
        # positive mini-batch sampling
        # the same num. of postive and negative samples are used in each batch
        pos_idx = np.random.choice(
            np.arange(num_pos), size=np.int(batch_size / 2))
        unlabeled_idx = np.random.choice(
            np.arange(num_unlabeled), size=np.int(batch_size / 2))
        # mini-batch samples
        train_batch = np.r_[pos_train[pos_idx], unlabeled_train[unlabeled_idx]]
        train_label = train_batch[:, 2]
        # define pscore score
        pscore_ = np.r_[pscore_pos_train[pos_idx],
                        pscore_unlabeled_train[unlabeled_idx]]
        train_score = pscore_ if ips else np.ones(batch_size)

        # update user-item latent factors and calculate training loss
        _, loss = sess.run([model.apply_grads, model.weighted_mse],
                           feed_dict={model.users: train_batch[:, 0],
                                      model.items: train_batch[:, 1],
                                      model.labels: np.expand_dims(train_label, 1),
                                      model.scores: np.expand_dims(train_score, 1)})
        train_loss_list.append(loss)
        # calculate ranking metrics
        if i % 25 == 0:
            u_emb, i_emb = sess.run(
                [model.user_embeddings, model.item_embeddings])
            ret = aoa_evaluator(user_embed=u_emb, item_embed=i_emb,
                                model_name=model_name, test=test, at_k=[5])
            test_dcg_list.append(ret.loc['DCG@5', model_name])
            test_map_list.append(ret.loc['MAP@5', model_name])
            test_recall_list.append(ret.loc['Recall@5', model_name])

    # save train and val loss curves.
    loss_path = Path(f'../logs/{model_name}/loss/')
    loss_path.mkdir(parents=True, exist_ok=True)
    np.save(file=str(loss_path / 'train.npy'), arr=train_loss_list)
    np.save(file=str(loss_path / 'dcg.npy'), arr=test_dcg_list)
    np.save(file=str(loss_path / 'recall.npy'), arr=test_recall_list)
    np.save(file=str(loss_path / 'map.npy'), arr=test_map_list)

    u_emb, i_emb = sess.run([model.user_embeddings, model.item_embeddings])

    sess.close()

    return u_emb, i_emb


class Trainer:
    """Trainer Class for ImplicitRecommender."""

    suffixes = ['all', 'rare']
    at_k = [1, 3, 5]
    rare_item_threshold = 500

    def __init__(self, max_iters: int = 1000, lam=1e-4, batch_size: int = 12,
                 eta: float = 0.1, model_name: str = 'wmf') -> None:
        """Initialize class."""
        path = f'../logs/{model_name}/tuning/best_params.json'
        best_params = json.loads(json.load(open(path, 'r')))
        self.dim = np.int(best_params['dim'])
        self.lam = lam
        self.weight = best_params['weight'] if model_name == 'wmf' else 1
        self.clip = best_params['clip'] if model_name == 'crmf' else 0
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.eta = eta
        self.model_name = model_name

    def run(self) -> None:
        """Train implicit recommenders."""
        train = np.load(f'../data/point/train.npy')
        test = np.load(f'../data/point/test.npy')
        pscore = np.load(f'../data/point/pscore.npy')
        item_freq = np.load(f'../data/point/item_freq.npy')
        num_users = np.int(train[:, 0].max() + 1)
        num_items = np.int(train[:, 1].max() + 1)

        tf.set_random_seed(12345)
        ops.reset_default_graph()
        sess = tf.Session()
        if self.model_name in ['rmf', 'wmf', 'crmf']:
            model = PointwiseRecommender(
                num_users=num_users, num_items=num_items, weight=self.weight,
                clip=self.clip, dim=self.dim, lam=self.lam, eta=self.eta)
            u_emb, i_emb = pointwise_trainer(
                sess, model=model, train=train, test=test, pscore=pscore,
                max_iters=self.max_iters, batch_size=2**self.batch_size,
                model_name=self.model_name)

        else:
            u_emb, i_emb = expomf_trainer(
                train=train, num_user=num_users, num_item=num_items,
                n_components=self.dim, lam=self.lam, model_name=self.model_name)

        # evaluate a given recommender
        ret_path = Path(f'../logs/{self.model_name}/results/')
        ret_path.mkdir(parents=True, exist_ok=True)
        test_rare = test[item_freq[test[:, 1]] <= self.rare_item_threshold]
        for test_, suffix in zip([test, test_rare], self.suffixes):
            results = aoa_evaluator(user_embed=u_emb, item_embed=i_emb,
                                    test=test_, model_name=self.model_name,
                                    at_k=self.at_k)

            results.to_csv(ret_path / f'aoa_{suffix}.csv')
