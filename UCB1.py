import numpy as np
class UCB1:
    def __init__(self, intents, arms):
        self.intents = intents
        self.arms = arms
        self.t = 0
        self.q = np.zeros((self.intents, self.arms))
        self.count = np.zeros((self.intents, self.arms))
        return

    def make_choice(self, intent):

        for i in range(self.arms):
            if self.count[intent, i] == 0:
                return i

        ucb_value = self.q[intent]
        ucb_value = ucb_value + np.sqrt((2*np.log(self.t))/self.count[intent])
        return np.argmax(ucb_value)


    def update_qvalue(self, intent, arm, reward):
        self.t += 1
        self.count[intent, arm] += 1
        n = self.count[intent, arm]
        self.q[intent, arm] = self.q[intent, arm] * ((n - 1) / n) + (reward / n)
        return

    def testing(self, intent):
        ucb_value = self.q[intent]
        ucb_value = ucb_value + np.sqrt((2 * np.log(self.t)) / self.count[intent])
        # for i in range(self.arms):
        #     if self.count[intent, i] > 0:
        #         ucb_value[i] += np.sqrt((2 * np.log(self.t)) / self.count[intent, i])

        return np.argmax(ucb_value)


if __name__ == '__main__':
    ucb = UCB1(10, 10)
    ucb.make_choice(10)
