"""
Codes for running the semi-synthetic experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""

import argparse
import sys
import json
import warnings

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

from data_generator import generate_real_data
from trainer import Trainer
from visualizer import plot_test_curves_with_ranking_metrics

parser = argparse.ArgumentParser()

parser.add_argument('model_name', type=str, choices=['wmf', 'expomf', 'crmf'])
parser.add_argument('--generate_data', action='store_true')


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()

    # hyper-parameters
    config = json.load(open('../config.json', 'rb'))
    eta = config['eta']
    batch_size = config['batch_size']
    max_iters = config['max_iters']
    threshold = config['threshold']
    lam = config['lam']
    generate_data = args.generate_data
    model_name = args.model_name

    # run simulations
    mlflow.set_experiment(f'yahoo')
    with mlflow.start_run() as run:
        if generate_data:
            generate_real_data(threshold=threshold)

        trainer = Trainer(batch_size=batch_size, max_iters=max_iters,
                          lam=lam, eta=eta, model_name=model_name)
        trainer.run()

        results_df = pd.read_csv(
            f'../logs/{model_name}/results/aoa_all.csv', index_col=0)
        mlflow.log_metric('MAP5', results_df.loc['MAP@5', model_name])
        mlflow.log_metric('Recall5', results_df.loc['Recall@5', model_name])
        mlflow.log_metric('DCG5', results_df.loc['DCG@5', model_name])

        if model_name != 'expomf':
            plot_test_curves_with_ranking_metrics(model=model_name)

        print('\n', '=' * 25, '\n')
        print(f'Finished Running {model_name}!')
        print('\n', '=' * 25, '\n')

        mlflow.log_param('threshold', threshold)
        mlflow.log_param('generate_data', generate_data)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('max_iters', max_iters)
        mlflow.log_param('lam', lam)
        mlflow.log_param('model_name', model_name)

        mlflow.log_artifacts(f'../logs/{model_name}/results/')

        if model_name != 'expomf':
            mlflow.log_artifacts(f'../plots/curves/{model_name}/')
