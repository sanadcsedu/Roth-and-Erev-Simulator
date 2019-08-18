import RothAndErevClass
import numpy as np
import ForRandomAgent
import matplotlib as plt


def plot(x_axis, y_axis, xstep=1.0, file=None):
    plt.xticks(np.arange(min(x_axis), max(x_axis), xstep))

    plt.plot(x_axis, y_axis, '-r')

    plt.ylabel('Accuracy')
    plt.xlabel('Number of iterations')
    plt.title('Accuracy Vs. Iterations')
    plt.legend(loc='best')
    if file is not None:
        plt.savefig('ML/%s' % file, bbox_inches='tight')
    plt.gcf().clear()


# rothanderev = RothAndErevClass.RothAndErevClass(5, 5, 0, 0, 0.0001, False)
# random_agent = ForRandomAgent.ForRandomAgent(10, 10, True)
# print(random_agent.make_choice())

experiments = iterations = 0
experiments = 1
iterations = 300
threshold = 0.8
cnt = 0
UserIntent = 10
QueryPerIntent = 10

accuracy = np.zeros(10)

pltx = []
plty = []

for exp in range(experiments):

    user = RothAndErevClass.RothAndErevClass(UserIntent, QueryPerIntent, 0, 0, 0.0001, False)
    dbms = RothAndErevClass.RothAndErevClass(QueryPerIntent, UserIntent, 0, 0, 0.0001, False)

    for intents in range(UserIntent):
        for itr in range(iterations):

            q = user.make_choice_wofails(intents, threshold)
            e = dbms.make_choice_wofails(q, threshold)
            #print("q = %d e = %d\n" %(q, e))

            #Add payoff 10 to the best intent-query pair matching
            if intents == q and intents == e:
                user.update_qtable(intents, e, 10)
                dbms.update_qtable(q, e, 10)

            # #Give Payoff 2 to the Adjacent Strategies
            # elif abs(intents - e) == 1:
            #     user.update_qtable(intents, e, 1)
            #     dbms.update_qtable(q, e, 1)

            else:
                user.remove_strategy(q)
                dbms.remove_strategy(e)

        trials = 10
        cnt = 0
        for itr in range(trials):
            q = user.testing(intents)
            e = dbms.testing(q)
            #print("%d %d\n" % (q, e))
            if intents == e and intents == q:
                cnt += 1
        accu = cnt / trials
        print(accu)

        # q = user.testing(intents)
        # e = dbms.testing(q)
        # if intents == e and intents == q:
        #     accuracy[intents] += 1

    # print()
    # print(np.sum(accuracy) / 10)
    # accuracy.fill(0)

# print(accuracy)
# print(np.sum(accuracy) / 50)







