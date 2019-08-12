import numpy as np

class EpsilonGreedy:

    def __init__(self, intent, arms, epsilon):
        self.intent = intent
        self.arms = arms
        self.epsilon = epsilon
        self.count = np.zeros((intent, arms))
        self.q = np.zeros((intent, arms))

    def make_choice(self, intent):
        if np.random.random() > self.epsilon: #Picking the best arm
            return np.argmax(self.q[intent])
        else: # Pick an arm randomly
            return np.random.randint(0, self.arms)

    def update_qvalue(self, intent, chosen_arm, reward):
        self.count[intent, chosen_arm] += 1
        n = self.count[intent, chosen_arm]

        # Recompute the estimated value of chosen arm using new reward
        self.q[intent, chosen_arm] = self.q[intent, chosen_arm] * ((n - 1) / n) + (reward / n)

    def testing(self, intent):
        return np.argmax(self.q[intent])
