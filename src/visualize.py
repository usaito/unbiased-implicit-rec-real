"""
Codes for visualizing results of the semi-synthetic experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""

import argparse
import sys
import warnings

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

from visualizer import plot_results_with_varying_K

parser = argparse.ArgumentParser()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()

    # run simulations
    mlflow.set_experiment(f'yahoo')
    with mlflow.start_run() as run:
        plot_results_with_varying_K()

        mlflow.log_artifacts(f'../plots/results/')
