import numpy as np


def softmax(x):
    """
    Customized Softmax function
    :param x: Probabilities between -1 and 1 (fire if 0 < prob < 1)
    :return: Softmax probabilities
    """
    x = np.array(x)
    e_x = np.exp(x)
    disregard_idxs = np.where(x < 0)
    probs = e_x / (e_x.sum(axis=0, keepdims=True) + 1.e-10)
    probs[disregard_idxs] = 0
    return probs
