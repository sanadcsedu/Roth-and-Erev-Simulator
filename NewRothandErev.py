import UCB1
import FixedStrategy
import RothAndErevClass
import EpsilonGreedy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def plot(x_axis, y_axis, _title, xstep=1.0, file=None):
    plt.xticks(np.arange(min(x_axis), max(x_axis)+1, xstep))
    plt.plot(x_axis, y_axis, '-r')

    plt.ylabel('Accuracy')
    plt.xlabel('Number of iterations x 100')
    plt.title(_title)
    plt.legend(loc='best')
    plt.show()
    plt.gcf().clear()


class NewRothandErev:
    def __init__(self, experiments=1, iterations=20000, UI=100, QI=100, use_prev_choice = False):
        self.experiments = experiments
        self.iterations = iterations
        self.UserIntent = UI
        self.QueryPerIntent = QI
        self.threshold = 0.7
        self.use_prev_choice = use_prev_choice

    def testing_module(self, user, dbms):
        trials = 10
        cnt = 0
        for intents in range(self.UserIntent):
            for itr in range(trials):
                q = user.testing(intents)
                e = dbms.testing(q)
                if intents == e:
                    cnt += 1

        return cnt / (self.UserIntent * trials)

    def dbms_ucb1(self):
        res = 0
        for exp in range(self.experiments):

            user = RothAndErevClass.RothAndErevClass(self.UserIntent, self.QueryPerIntent, 0, 0, 0.1, False)
            dbms = UCB1.UCB1(self.QueryPerIntent, self.UserIntent)

            for intents in tqdm(range(self.UserIntent)):
                for itr in range(self.iterations):

                    q = user.make_choice_wofails(intents, self.threshold)
                    e = dbms.make_choice(q)
                    if intents == e:
                        user.update_qtable(intents, q, 10, False)
                        dbms.update_qvalue(q, e, 10)
                    else:
                        user.remove_strategy(q)
                        dbms.update_qvalue(q, e, 1)

            res += self.testing_module(user, dbms)
        return res / self.experiments

    def dbms_fixed(self):
        res = 0
        for exp in range(self.experiments):

            user = RothAndErevClass.RothAndErevClass(self.UserIntent, self.QueryPerIntent, 0, 0, 0.1, False)
            dbms = FixedStrategy.FixedStrategy(self.QueryPerIntent, self.UserIntent, 1)
            for intents in tqdm(range(self.UserIntent)):

                for itr in range(self.iterations):

                    q = user.make_choice_wofails(intents, self.threshold)
                    e = dbms.make_choice(q)
                    if intents == e:
                        user.update_qtable(intents, q, 10, False)
                    else:
                        user.remove_strategy(q)
            res += self.testing_module(user, dbms)
        return res / self.experiments

    def dbms_EpsilonGreedy(self):
        res = 0
        for exp in range(self.experiments):

            user = RothAndErevClass.RothAndErevClass(self.UserIntent, self.QueryPerIntent, 0, 0, 0.0001, False)
            dbms = EpsilonGreedy.EpsilonGreedy(self.UserIntent, self.QueryPerIntent, 0.5)

            for intents in tqdm(range(self.UserIntent)):
                for itr in range(self.iterations):

                    q = user.make_choice_wofails(intents, self.threshold)
                    e = dbms.make_choice(q)
                    if intents == e:
                        user.update_qtable(intents, q, 10)
                        dbms.update_qvalue(q, e, 10)
                    else:
                        user.remove_strategy(q)
                        dbms.update_qvalue(q, e, 1)

            res += self.testing_module(user, dbms)
        return res / self.experiments

    def dbms_RothAndErev(self):
        res = 0
        for exp in range(self.experiments):

            user = RothAndErevClass.RothAndErevClass(self.UserIntent, self.QueryPerIntent, 0, 0, 0.01, False)
            dbms = RothAndErevClass.RothAndErevClass(self.QueryPerIntent, self.UserIntent, 0, 0, 0.0001, False)
            for intents in tqdm(range(self.UserIntent)):

                for itr in range(self.iterations):

                    q = user.make_choice_wofails(intents, self.threshold)
                    e = dbms.make_choice_wofails(q, self.threshold)

                    if intents == e:
                        user.update_qtable(intents, q, 10, False)
                        dbms.update_qtable(q, e, 10, True)
                    else:
                        user.remove_strategy(q)
                        dbms.remove_strategy(e)
            res += self.testing_module(user, dbms)
        return res / self.experiments

if __name__ == '__main__':
    x = []
    y = []
    cnt = 1
    for itr in range(100, 1001, 100):
        tester = NewRothandErev(5, itr, 100, 100)
    # for itr in range(1000, 20001, 1000):
    #     tester = NewRothandErev(1, itr, 100, 100)
        x.append(cnt)
        #y.append(tester.dbms_ucb1())
        #y.append(tester.dbms_fixed())
        #y.append(tester.dbms_EpsilonGreedy())
        y.append(tester.dbms_RothAndErev())
        print(cnt)
        cnt += 1

    print(x)
    print(y)
    title = 'User [Modified Roth and Erev] vs DBMS [Roth and Erev]'
    plot(x, y, title, xstep=1.0, file=None)

