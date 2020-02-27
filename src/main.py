"""
Codes for running the semi-synthetic experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""

import argparse
import sys
import yaml
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from data.preprocessor import preprocess_dataset
from trainer import Trainer
from visualize import plot_test_curves_with_ranking_metrics

parser = argparse.ArgumentParser()

parser.add_argument('model_name', type=str, choices=['wmf', 'expomf', 'crmf'])
parser.add_argument('--preprocess_data', action='store_true')


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()

    # hyper-parameters
    config = yaml.safe_load(open('../config.yaml', 'rb'))
    eta = config['eta']
    batch_size = config['batch_size']
    max_iters = config['max_iters']
    threshold = config['threshold']
    lam = config['lam']
    model_name = args.model_name

    # run simulations
    if args.preprocess_data:
        preprocess_dataset(threshold=threshold)

    trainer = Trainer(batch_size=batch_size, max_iters=max_iters,
                      lam=lam, eta=eta, model_name=model_name)
    trainer.run()

    if model_name != 'expomf':
        plot_test_curves_with_ranking_metrics(model=model_name)

    print('\n', '=' * 25, '\n')
    print(f'Finished Running {model_name}!')
    print('\n', '=' * 25, '\n')
