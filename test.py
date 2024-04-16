import numpy as np


def test(a: int):
    for i in a:
        print(i)


a = np.array([1, 2, 3, 4, 5])
test(a)
