import numpy as np


def softmax(x):
    # disregarding logits less than 0
    e_x = np.exp(np.maximum(x, 0))
    return e_x / (e_x.sum(axis=0, keepdims=True) + 1.e-10)
