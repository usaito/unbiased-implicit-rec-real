"""
Codes for preprocessing real-world datasets used in the experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""
import codecs
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.model_selection import train_test_split


def preprocess_dataset(threshold: int = 4) -> Tuple:
    """Load and Preprocess datasets."""
    # load dataset.
    col = {0: 'user', 1: 'item', 2: 'rate'}
    with codecs.open(f'../data/train.txt', 'r', 'utf-8', errors='ignore') as f:
        data_train = pd.read_csv(f, delimiter='\t', header=None)
        data_train.rename(columns=col, inplace=True)
    with codecs.open(f'../data/test.txt', 'r', 'utf-8', errors='ignore') as f:
        data_test = pd.read_csv(f, delimiter='\t', header=None)
        data_test.rename(columns=col, inplace=True)
    num_users, num_items = data_train.user.max(), data_train.item.max()
    for _data in [data_train, data_test]:
        _data.user, _data.item = _data.user - 1, _data.item - 1
        # binalize rating.
        _data.rate[_data.rate < threshold] = 0
        _data.rate[_data.rate >= threshold] = 1
    # train-val-test, split
    train, test = data_train.values, data_test.values
    train, val = train_test_split(train, test_size=0.1, random_state=12345)
    # estimate pscore
    _, item_freq = np.unique(train[train[:, 2] == 1, 1], return_counts=True)
    pscore = (item_freq / item_freq.max()) ** 0.5
    # only positive data
    train = train[train[:, 2] == 1, :2]
    val = val[val[:, 2] == 1, :2]

    # creating training data
    all_data = pd.DataFrame(
        np.zeros((num_users, num_items))).stack().reset_index()
    all_data = all_data.values[:, :2]
    unlabeled_data = np.array(
        list(set(map(tuple, all_data)) - set(map(tuple, train))), dtype=int)
    train = np.r_[np.c_[train, np.ones(train.shape[0])],
                  np.c_[unlabeled_data, np.zeros(unlabeled_data.shape[0])]]
    # save datasets
    path = Path(f'../data/point')
    path.mkdir(parents=True, exist_ok=True)
    np.save(str(path / 'train.npy'), arr=train.astype(np.int))
    np.save(str(path / 'val.npy'), arr=val.astype(np.int))
    np.save(str(path / 'test.npy'), arr=test.astype(np.int))
    np.save(str(path / 'pscore.npy'), arr=pscore)
    np.save(str(path / 'item_freq.npy'), arr=item_freq)
