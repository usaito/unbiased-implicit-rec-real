"""
Recommender models used for the semi-synthetic experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""
from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf


class AbstractRecommender(metaclass=ABCMeta):
    """Abstract base class for evaluator class."""

    @abstractmethod
    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        raise NotImplementedError()

    @abstractmethod
    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        raise NotImplementedError()

    @abstractmethod
    def create_losses(self) -> None:
        """Create the losses."""
        raise NotImplementedError()

    @abstractmethod
    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        raise NotImplementedError()


class PointwiseRecommender(AbstractRecommender):
    """Implicit Recommenders based on pointwise approach."""

    def __init__(self, num_users: np.array, num_items: np.array,
                 dim: int, lam: float, eta: float, weight: float = 1.0, clip: float = 0) -> None:
        """Initialize Class."""
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.lam = lam
        self.eta = eta
        self.weight = weight
        self.clip = clip

        # Build the graphs
        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.users = tf.placeholder(tf.int32, [None], name='user_placeholder')
        self.items = tf.placeholder(tf.int32, [None], name='item_placeholder')
        self.scores = tf.placeholder(
            tf.float32, [None, 1], name='score_placeholder')
        self.labels = tf.placeholder(
            tf.float32, [None, 1], name='label_placeholder')

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope('embedding_layer'):
            # initialize user-item matrices and biases
            self.user_embeddings = tf.get_variable(
                'user_embeddings', shape=[self.num_users, self.dim],
                initializer=tf.contrib.layers.xavier_initializer())
            self.item_embeddings = tf.get_variable(
                'item_embeddings', shape=[self.num_items, self.dim],
                initializer=tf.contrib.layers.xavier_initializer())

            # lookup embeddings of current batch
            self.u_embed = tf.nn.embedding_lookup(
                self.user_embeddings, self.users)
            self.i_embed = tf.nn.embedding_lookup(
                self.item_embeddings, self.items)

        with tf.variable_scope('prediction'):
            self.logits = tf.reduce_sum(
                tf.multiply(self.u_embed, self.i_embed), 1)
            self.preds = tf.sigmoid(tf.expand_dims(
                self.logits, 1), name='sigmoid_prediction')

    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('losses'):
            # define the unbiased binary cross entropy loss in Eq. (9).
            scores = tf.clip_by_value(
                self.scores, clip_value_min=self.clip, clip_value_max=1.0)
            self.weighted_mse = tf.reduce_sum(
                (self.labels / scores) * tf.square(1. - self.preds) +
                self.weight * (1 - self.labels / scores) * tf.square(self.preds)) / \
                tf.reduce_sum(self.labels + self.weight * (1 - self.labels))

            # add the L2-regularizer terms.
            reg_term_embeds = tf.nn.l2_loss(self.user_embeddings) \
                + tf.nn.l2_loss(self.item_embeddings)
            self.loss = self.weighted_mse + self.lam * reg_term_embeds

    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        with tf.name_scope('optimizer'):
            # set Adam Optimizer.
            self.apply_grads = tf.train.AdamOptimizer(
                learning_rate=self.eta).minimize(self.loss)
