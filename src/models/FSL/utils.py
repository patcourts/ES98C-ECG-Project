
"""A collection of utility functions and objects for Tensorflow models."""

__all__ = ['reduce_tensor', 'reshape_query', 'proto_dist', 'LinearFusion']

import tensorflow as tf
from keras import Layer


def reduce_tensor(X):
    """Reduce sample embedding."""
    return tf.reduce_mean(X, axis=1)


def reshape_query(X):
    """Reshape query embedding."""
    return tf.reshape(X, [-1, tf.shape(X)[-1]])  # type: ignore


def proto_dist(X):
    """Prototypical distance."""
    Xs, Xq = X
    Xq_r = tf.reduce_sum(Xq ** 2, axis=1, keepdims=True)
    Xs_r = tf.reduce_sum(Xs ** 2, axis=1, keepdims=True)
    qdots = tf.matmul(Xq, tf.transpose(Xs))
    return tf.nn.softmax(-(tf.sqrt(Xq_r + tf.transpose(Xs_r) - 2 * qdots)))


class LinearFusion(Layer):
    """Linear Fusion Layer.

    Parameters
    ----------
    shot : int
        number of samples in n-way, k-shot episodes.
    output_dim : tuple
        dimensions of output tensor
    """

    def __init__(self, shot, output_dim, **kwargs):
        self.output_dim = output_dim
        self.shot = shot
        super().__init__(**kwargs)

    def build(self, input_shape):
        """keras.Layer build override."""
        self.sigma = self.add_weight(
            name='sigma',
            shape=(self.shot, self.output_dim),
            initializer='random_uniform',
            trainable=True
        )

    def call(self, x1, x2):
        """keras.Layer call override."""
        return self.sigma * x1 + (1 - self.sigma) * x2