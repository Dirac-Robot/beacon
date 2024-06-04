from typing import Iterable


def is_seq(x):
    return isinstance(x, Iterable) and not isinstance(x, str)


def get_all(iterable, key):
    return map(lambda x: x[key], iterable)
