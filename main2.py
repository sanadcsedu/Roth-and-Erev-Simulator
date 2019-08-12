import RothAndErevClass
import FixedStrategy
import numpy as np
import ForRandomAgent
import matplotlib as plt
from tqdm import tqdm

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
iterations = 0
threshold = 0.8
cnt = 0
UserIntent = 10
QueryPerIntent = 10

accuracy = np.zeros(10)

pltx = []
plty = []
total = 0
for exp in range(experiments):

    user = RothAndErevClass.RothAndErevClass(UserIntent, QueryPerIntent, 0, 0, 0.0001, False)
    dbms = FixedStrategy.FixedStrategy(QueryPerIntent, UserIntent, 0.6)

    for intents in range(UserIntent):
        for itr in tqdm(range(iterations)):

            q = user.make_choice(intents, threshold)
            e = dbms.make_choice(q)
            #print("De %d %d\n" % (q, e))

            #Add payoff 10 to the best intent-query pair matching
            if intents == e:
                user.update_qtable(intents, q, 10)

        trials = 10
        cnt = 0
        for itr in range(trials):
            q = user.testing(intents)
            e = dbms.make_choice(q)
            print("%d %d\n" % (q, e))
            if intents == e:
                cnt += 1
        accu = cnt / trials
        total += cnt
        print(accu)

    print(total)
    print(total/intents*trials)





