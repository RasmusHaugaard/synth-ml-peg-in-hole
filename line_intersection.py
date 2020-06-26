import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


def two_dimensional():
    while True:
        l1 = np.random.uniform(-1, 1, 3)
        v = (1, .5)
        l1 = -v[1], v[0], 0
        l2 = np.random.uniform(-1, 1, 3)
        p = np.cross(l1, l2)

        x = np.linspace(-10, 10)

        for name, y in zip(['l1', 'l2'], [(l[0] * x + l[2]) / -l[1] for l in (l1, l2)]):
            plt.plot(x, y, label=name)

        plt.legend()
        plt.scatter(*(p[:2, None] / p[2]), c='k', zorder=3, label='a-b intersection')
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.gca().set_aspect(1)
        plt.show()


def line_eq(p0, p1):
    p0, p1 = np.asarray(p0), np.asarray(p1)
    assert p0.shape == p1.shape
    assert len(p0.shape) == 1
    d = p0.shape[0]
    assert d > 1
    A = np.zeros((d, d))
    A[np.triu_indices(d, 1)] = 1
    A[np.tril_indices(d, -1)] = -1
    l = (p1 - p0) @ A
    l = np.concatenate((l, [-np.sum(l * p0)]))
    return l / np.linalg.norm(l)


def three_dimensional():
    pass


if __name__ == '__main__':
    d, n = 3, int(1e3)
    for _ in range(n):
        p0, p1 = np.random.randn(2, d)
        l = line_eq(p0, p1)
        for p in p0, p1:
            assert abs(np.sum(l[:d] * p) + l[d]) < 1e-10
