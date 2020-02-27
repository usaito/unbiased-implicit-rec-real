"""
Codes for visualizing results of the semi-synthetic experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from plotly.graph_objs import Bar, Figure, Histogram, Layout, Scatter
from plotly.offline import plot

all_models = ['wmf', 'expomf', 'crmf']
K = [1, 3, 5]
metrics = ['DCG', 'Recall', 'MAP']
all_names = ['WMF', 'ExpoMF', 'Rel-MF with Clipping']


def plot_results_with_varying_K() -> None:
    """Plot results with varying K."""
    # load results
    path = Path(f'../plots/results/')
    path.mkdir(parents=True, exist_ok=True)
    _load_and_save_results()
    for data in ['rare', 'all']:
        for met in metrics:
            col = [f'{met}@{k}' for k in K]
            ret_file = f'../logs/overall/results/ranking_{data}.csv'
            ret = pd.read_csv(ret_file, index_col=0).T[col]
            scatter_list = [Scatter(x=K, y=np.array(ret)[i, :],
                                    marker=dict(size=20), line=dict(width=6), name=name)
                            for i, name in enumerate(all_names)]

            layout_results_with_varying_K.title.text = f'{met} ({data})'
            plot(Figure(data=scatter_list, layout=layout_test_curves_with_ranking_metrics),
                 filename=str(path / f'{met}_{data}.html'), auto_open=False)


def plot_test_curves_with_ranking_metrics(model: str) -> None:
    """Plot test curves."""
    scatter_list = []
    path = Path(f'../plots/curves/{model}/')
    path.mkdir(parents=True, exist_ok=True)
    dcg = np.load(f'../logs/{model}/loss/dcg.npy')
    map = np.load(f'../logs/{model}/loss/map.npy')
    recall = np.load(f'../logs/{model}/loss/recall.npy')

    scatter_list.append(Scatter(x=np.arange(0, len(dcg) * 25, 25),
                                y=dcg, name='DCG@5', opacity=0.8,
                                mode='lines', line=dict(width=6)))
    scatter_list.append(Scatter(x=np.arange(0, len(recall) * 25, 25),
                                y=recall, name='Recall@5', opacity=0.8,
                                mode='lines', line=dict(width=6)))
    scatter_list.append(Scatter(x=np.arange(0, len(map) * 25, 25),
                                y=map, name='MAP@5', opacity=0.8,
                                mode='lines', line=dict(width=6)))

    layout_test_curves_with_ranking_metrics.title.text = f'Test Curves of {model}'
    plot(Figure(data=scatter_list, layout=layout_test_curves_with_ranking_metrics),
         filename=str(path / 'test_curves_with_ranking_metrics.html'), auto_open=False)


def _load_and_save_results() -> None:
    """Load and save experimental results."""
    path = Path(f'../logs/overall/results/')
    path.mkdir(parents=True, exist_ok=True)
    for suffix in ['rare', 'all']:
        aoa_list = []
        for model in all_models:
            ret_file = f'../logs/{model}/results/aoa_{suffix}.csv'
            aoa_list.append(pd.read_csv(ret_file, index_col=0))
        result_df = pd.concat([aoa_list[i]
                               for i in np.arange(len(all_models))], 1)
        for model in all_models:
            result_df['crmf/' + model] = result_df['crmf'] / result_df[model]
        result_df.to_csv(str(path / f'ranking_{suffix}.csv'))


layout_test_curves_with_ranking_metrics = Layout(
    title=dict(font=dict(size=40), x=0.5, xanchor='center', y=0.99),
    paper_bgcolor='rgb(255,255,255)', plot_bgcolor='rgb(235,235,235)',
    width=1200, height=800,
    xaxis=dict(title='iterations', titlefont=dict(size=30),
               tickfont=dict(size=18), gridcolor='rgb(255,255,255)'),
    yaxis=dict(tickfont=dict(size=18), gridcolor='rgb(255,255,255)'),
    legend=dict(bgcolor='rgb(245,245,245)', x=0.01, xanchor='left',
                orientation='h', y=0.99, yanchor='top', font=dict(size=30)),
    margin=dict(l=80, t=50, b=60))

layout_results_with_varying_K = Layout(
    title=dict(font=dict(size=40), x=0.5, xanchor='center', y=0.99),
    paper_bgcolor='rgb(255,255,255)', plot_bgcolor='rgb(235,235,235)',
    width=1200, height=800,
    xaxis=dict(title='K', titlefont=dict(size=30),
               tickmode='array', tickvals=K, ticktext=K,
               tickfont=dict(size=20), gridcolor='rgb(255,255,255)'),
    yaxis=dict(tickfont=dict(size=20), gridcolor='rgb(255,255,255)'),
    legend=dict(bgcolor='rgb(245,245,245)', x=0.01, xanchor='left',
                orientation='h', y=0.99, yanchor='top', font=dict(size=32)),
    margin=dict(l=60, t=50, b=60))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    plot_results_with_varying_K()
