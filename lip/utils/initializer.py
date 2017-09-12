import tensorflow as tf
import numpy as np


def xavier_init(fan_in: int, fan_out: int, const: int= 1) -> tf.Tensor:
    """ Xavier initialization of network weights.
    https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    https://gist.github.com/blackecho/3a6e4d512d3aa8aa6cf9
    :param fan_in: fan in of the network (n_features)
    :param fan_out: fan out of the network (n_components)
    :param const: multiplicative constant (4 for sigmoid, 1 for tanh)
    """
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high)
