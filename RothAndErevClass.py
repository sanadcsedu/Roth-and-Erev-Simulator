import numpy as np

class RothAndErevClass:

    def __init__(self, n, strategies, cutoff = 0, experimentation = 0, forgetting = 0, random = False):
        self.n = n
        self.strategies = strategies
        self.cutoff = cutoff
        self.experimentation = experimentation
        self.forgetting = forgetting

        if random:
            self.q = np.random.binomial(strategies, 0.5, (self.n, self.strategies))
            #self.q = np.random.random((self.n, self.strategies))
        else:
            self.q = np.full((self.n, self.strategies), 1)

        self.p = np.zeros((self.n, self.strategies))
        for i in range(self.n):
            self.p[i, :] = self.q[i, :] / np.sum(self.q[i])

    def show(self):
        print(self.q)
        print()
        print(self.p)

    def choose(self, n, k, payoff):

        #Applying Cutoff Parameter [Prevent low probabilistic outcomes to influence the outcome]

        if(self.cutoff > 0):

            self.q[n, k] += payoff
            self.p[n, :] = self.q[n, :] / np.sum(self.q[n])

            for i in range (self.strategies):
                if self.p[n, i] < self.cutoff:
                    self.q[n, i] = self.p[n, i] = 0
            self.p[n, :] = self.q[n, :] / np.sum(self.q[n])

        #Applying Persistent local experimentation [Preventing probability of Adjacent strategies to reach to 0]

        if self.experimentation > 0:

            self.q[n, k] += payoff * (1 - self.experimentation)
            #Add (self.experimentation * payoff) to Adjacent Strategies
            #implementation depends on Providing Adjacent Strategies
            self.p[n, :] = self.q[n, :] / np.sum(self.q[n])

        #Applying Gradual forgetting
        if self.forgetting > 0:
            self.q[n, k] += payoff
            self.p[n, :] = self.q[n, :] / np.sum(self.q[n])
            self.q = self.q * (1 - self.forgetting)





