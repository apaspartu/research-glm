import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    """Derivative of `sigmoid` function."""
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy(y_predicted: np.ndarray, y_observed: np.ndarray) -> float:
    return float(-np.sum(np.multiply(y_observed, np.log(y_predicted))))


def d_cross_entropy(y_predicted: np.ndarray, y_observed: np.ndarray) -> np.ndarray:
    """Derivative of `cross-entropy` with respect to `softmax`,
       which in turn with respect to `y_predicted`."""
    # return (softmax(y_predicted) - y_observed) * y_observed
    return softmax(y_predicted) - y_observed


def l2_norm(x: np.ndarray) -> float:
    return np.sqrt(np.sum(x ** 2))


def clip_gradient(g: np.ndarray, theta: [int, float]) -> np.ndarray:
    return min(1, (theta / l2_norm(g))) * g


def one_hot(vector: np.ndarray):
    index = argmax(vector)
    one_hot_vec = np.zeros(shape=vector.size)
    one_hot_vec[index] = 1.
    return one_hot_vec.reshape((1, vector.size))


def argmax(vector: np.ndarray, k=1):
    """Find indices of the largest `k` values of the vector."""
    if k < 1:
        raise ValueError('k must be positive integer')

    array = vector.reshape(vector.size).tolist()
    sorted_array = list(sorted(array))
    indices = []
    for feature in sorted_array[len(array) - k:]:
        indices.append(array.index(feature))
    return indices if len(indices) > 1 else indices[0]
