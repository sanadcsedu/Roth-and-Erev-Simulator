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

        self.options = []
        for i in range(strategies):
            self.options.append(i)

    def show(self):
        np.set_printoptions(precision=2, suppress=True)
        # print(self.p)
        print(self.options)
        print(self.p)

    def remove_strategy(self, x):
        if 2 * len(self.options) > self.strategies and self.options.count(x) > 0:
            self.options.remove(x)
        return

    def update_qtable(self, n, k, payoff, basic = True):

        #Running the Basic Model
        self.q[n, k] += payoff
        self.p[n, :] = self.q[n, :] / np.sum(self.q[n])

        if basic:
            return

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

    def make_choice(self, n, threshold):

        max_prob = np.max(self.q[n, :])
        if max_prob < threshold:
            max_idx = np.argwhere(self.q[n, :] == np.amax(self.q[n, :]))
            idx_lst = max_idx.flatten().tolist()
            return np.random.choice(idx_lst)
        else:
            return np.random.randint(0, self.strategies)

    def make_choice_wofails(self, n, threshold):

        candidates = []
        max_prob = np.max(self.q[n, :])
        max_idx = np.argwhere(self.q[n, :] == np.amax(self.q[n, :]))
        idx_lst = max_idx.flatten().tolist()
        candidates.append(np.random.choice(idx_lst))
        if max_prob < threshold:
            candidates.append(np.random.choice(self.options))

        return np.random.choice(candidates)

    def testing(self, n):
        max_idx = np.argwhere(self.q[n, :] == np.amax(self.q[n, :]))
        idx_lst = max_idx.flatten().tolist()
        return np.random.choice(idx_lst)



