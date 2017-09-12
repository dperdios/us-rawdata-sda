import numpy as np
import scipy.linalg
from typing import Tuple, Optional


def create_gaussian_rip_matrix(size: Tuple[int, int], seed: Optional = None) -> np.ndarray:
    """
    Create a Gaussian matrix satisfying the Restricted Isometry Property (RIP).
    See: H. Rauhut - Compressive Sensing and Structured Random Matrices
    :param size: list or tuple of ints
    :param seed: int or array_like, optional
    :return: matrix: array, shape = (m, n)
    """
    m, n = size
    mean = 0.0
    stdev = 1 / np.sqrt(m)
    prng = np.random.RandomState(seed=seed)
    matrix = prng.normal(loc=mean, scale=stdev, size=size)

    return matrix


def create_gaussian_orth_matrix(size: Tuple[int, int], seed: Optional = None) -> np.ndarray:
    """
    Create an orthonormalized Gaussian matrix satisfying the Restricted Isometry Property (RIP).
    See: H. Rauhut - Compressive Sensing and Structured Random Matrices
    :param size: int or tuple of ints, optional
    :param seed: int or array_like, optional
    :return: matrix: array, shape = (m, n)
    """
    m, n = size
    mean = 0.0
    stdev = 1 / np.sqrt(m)
    prng = np.random.RandomState(seed=seed)
    matrix_orth = prng.normal(loc=mean, scale=stdev, size=(n, n))
    matrix_orth = scipy.linalg.orth(matrix_orth)
    matrix = matrix_orth[0:m, :]

    return matrix


def create_bernoulli_rip_matrix(size: Tuple[int, int], seed: Optional = None) -> np.ndarray:
    """
    Create a Bernoulli matrix satisfying the Restricted Isometry Property (RIP).
    See: H. Rauhut - Compressive Sensing and Structured Random Matrices
    :param size: int or tuple of ints, optional. Default is None
    :param seed: int or array_like, optional
    :return: matrix: array, shape = (m, n)
    """
    m, n = size
    prng = np.random.RandomState(seed=seed)
    matrix = prng.randint(low=0, high=2, size=size).astype('float')  # gen 0, +1 sequence
    # astype('float') required to use the true divide (/=) which follows
    matrix *= 2
    matrix -= 1
    matrix /= np.sqrt(m)

    return matrix


def create_cs_measurement_model(type: str, input_dim: int, compression_percent: int,
                                seed: Optional = None) -> np.ndarray:
    """
    Create standard compressed sensing measurement model of shape M x N
    """
    if seed is None:
        seed = 1234567890

    n = input_dim

    m = round((1 - compression_percent/100) * n)

    if type.lower() == 'gaussian-rip':
        measurement_model = create_gaussian_rip_matrix(size=(m, n), seed=seed)
    elif type.lower() == 'gaussian-orth':
        measurement_model = create_gaussian_orth_matrix(size=(m, n), seed=seed)
    elif type.lower() == 'bernoulli-rip':
        measurement_model = create_bernoulli_rip_matrix(size=(m, n), seed=seed)
    elif type.lower() == 'identity':
        if m is not n:
            TypeError('\'Identity\' measurement model only possible without compression')
        measurement_model = np.eye(n)
    else:
        raise NameError('Undefined measurement model type')

    return measurement_model
