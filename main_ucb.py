import UCB1
import FixedStrategy
import RothAndErevClass
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def plot(x_axis, y_axis, xstep=1.0, file=None):
    plt.xticks(np.arange(min(x_axis), max(x_axis)+1, xstep))

    plt.plot(x_axis, y_axis, '-r')

    plt.ylabel('Accuracy')
    plt.xlabel('Number of iterations x 1000')
    plt.title('User [Roth and Erev] vs DBMS [Roth and Erev]')
    plt.legend(loc='best')
    #if file is not None:
    #    plt.savefig('ML/%s' % file, bbox_inches='tight')
    plt.show()
    plt.gcf().clear()



class main_ucb:
    def __init__(self, experiments=1, iterations=20000, UI=100, QI=100):
        self.experiments = experiments
        self.iterations = iterations
        self.UserIntent = UI
        self.QueryPerIntent = QI
        self.original_mapping = np.zeros((UI, QI))
        self.threshold = 0.7

    def create_original_mapping(self):
        candid = []
        for i in range(self.QueryPerIntent):
            candid.append(i)
        for _intent in range(self.UserIntent):
            mkch = np.random.choice(candid)
            candid.remove(mkch)
            self.original_mapping[_intent, mkch] = 1

    def testing_module(self, user, dbms, ucbucb):
        trials = 1
        cnt = 0
        for intents in range(self.UserIntent):
            for itr in range(trials):
                q = user.testing(intents)
                e = dbms.testing(q)
                # print("%d %d" % (q,e))
                if ucbucb == 1:
                    if intents == e:
                        cnt += 1
                else:
                    if (intents == e) and (self.original_mapping[intents, q] == 1):
                        cnt += 1


        #print(cnt)
        #print(cnt / (self.UserIntent * trials))
        return cnt / (self.UserIntent * trials)

    def ucb_vs_fixed(self):
        for exp in range(self.experiments):

            user = UCB1.UCB1(self.UserIntent, self.QueryPerIntent)
            dbms = FixedStrategy.FixedStrategy(self.QueryPerIntent, self.UserIntent, 0.6)

            # For Printing DBMS Query-to-Intent Mapping
            # for query in range(QueryPerIntent):
            #     _intent = dbms.make_choice(query)
            #     print("Query " + str(query) + " intent " + str(_intent))

            for intents in tqdm(range(self.UserIntent)):
                for itr in range(self.iterations):

                    q = user.make_choice(intents)
                    e = dbms.make_choice(q)

                    # Add payoff 10 to the best intent-query pair matching
                    if intents == e:
                        user.update_qvalue(intents, q, 10)
                    else:
                        user.update_qvalue(intents, q, 1)

        return self.testing_module(user, dbms, 1)

    def ucb1_vs_ucb1(self):
        for exp in range(self.experiments):

            user = UCB1.UCB1(self.UserIntent, self.QueryPerIntent)
            dbms = UCB1.UCB1(self.QueryPerIntent, self.UserIntent)
            #print(self.original_mapping)
            for intents in tqdm(range(self.UserIntent)):

                for itr in range(self.iterations):

                    q = user.make_choice(intents)
                    e = dbms.make_choice(q)

                    if (intents == e) and (self.original_mapping[intents, q] == 1):
                        user.update_qvalue(intents, q, 10)
                        dbms.update_qvalue(q, e, 10)
                    else:
                        user.update_qvalue(intents, q, 1)
                        dbms.update_qvalue(q, e, 1)

        return self.testing_module(user, dbms, 0)

    def RothErev_vs_ucb1(self):
        for exp in range(self.experiments):

            user = RothAndErevClass.RothAndErevClass(self.UserIntent, self.QueryPerIntent, 0, 0, 0.0001, False)
            dbms = UCB1.UCB1(self.QueryPerIntent, self.UserIntent)
            #print(self.original_mapping)
            for intents in tqdm(range(self.UserIntent)):

                for itr in range(self.iterations):

                    q = user.make_choice_wofails(intents, self.threshold)
                    e = dbms.make_choice(q)

                    if (intents == e) and (self.original_mapping[intents, q] == 1):
                        user.update_qtable(intents, q, 10)
                        dbms.update_qvalue(q, e, 10)
                    else:
                        user.remove_strategy(q)
                        dbms.update_qvalue(q, e, 1)

        return self.testing_module(user, dbms, 0)

    def RothErev_vs_RothErev(self):
        for exp in range(self.experiments):

            user = RothAndErevClass.RothAndErevClass(self.UserIntent, self.QueryPerIntent, 0, 0, 0.0001, False)
            dbms = RothAndErevClass.RothAndErevClass(self.QueryPerIntent, self.UserIntent, 0, 0, 0.0001, False)
            #print(self.original_mapping)
            for intents in tqdm(range(self.UserIntent)):

                for itr in range(self.iterations):

                    q = user.make_choice_wofails(intents, self.threshold)
                    e = dbms.make_choice_wofails(q, self.threshold)

                    if (intents == e) and (self.original_mapping[intents, q] == 1):
                        user.update_qtable(intents, q, 10)
                        dbms.update_qtable(q, e, 10)
                    else:
                        user.remove_strategy(q)
                        dbms.remove_strategy(e)

        return self.testing_module(user, dbms, 0)



if __name__ == '__main__':
    x = []
    y = []
    cnt = 1
    for itr in range(1000, 20001, 1000):
        ucb_tester = main_ucb(1, itr, 100, 100)
    # for itr in range(100, 1001, 100):
    #     ucb_tester = main_ucb(1, itr, 10, 10)

        ucb_tester.create_original_mapping()

        x.append(cnt)
        y.append(ucb_tester.RothErev_vs_RothErev())
        print(cnt)
        cnt += 1

    plot(x, y, xstep=1.0, file=None)

    # ucb_tester = main_ucb()
    # #ucb_tester.ucb_vs_fixed()
    # ucb_tester.create_original_mapping()
    # ucb_tester.ucb1_vs_ucb1()
