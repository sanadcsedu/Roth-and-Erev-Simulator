import numpy as np


class ForRandomAgent:

    def __init__(self, n, strategies, random=True):
        self.n = n
        self.strategies = strategies

        if random:
            self.q = np.random.binomial(strategies, 0.5, (self.n, self.strategies))
        else:
            self.q = np.full((self.n, self.strategies), 1)

        self.p = np.zeros((self.n, self.strategies))
        for i in range(self.n):
            self.p[i, :] = self.q[i, :] / np.sum(self.q[i])

    def make_choice(self):
        return np.random.randint(0, self.strategies)

    def update_table(self, n, k, payoff):
        self.q[n, k] += payoff
        self.p[n, :] = self.q[n, :] / np.sum(self.q[n])

    def show(self):
        np.set_printoptions(precision=2, suppress=True)
        print(self.p)
# import numpy as np
# np.random.seed(30)
# a = np.zeros((10,), dtype = int)

# for i in range(30):
# 	x = np.random.randint(0, 10)
# 	print(x)
# 	a[x]+=1
#
# for i, val in enumerate(a):
# 	print(i, val)
