import numpy as np


def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump
