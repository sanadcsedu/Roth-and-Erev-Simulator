import RothAndErevClass
import FixedStrategy
import numpy as np
import EpsilonGreedy
from tqdm import tqdm

experiments = iterations = 0
experiments = 1
iterations = 10
threshold = 0.8
cnt = 0
UserIntent = 10
QueryPerIntent = 10

accuracy = np.zeros(10)

pltx = []
plty = []
total = 0
for exp in range(experiments):

    user = EpsilonGreedy.EpsilonGreedy(UserIntent, QueryPerIntent, 0.5)
    dbms = FixedStrategy.FixedStrategy(QueryPerIntent, UserIntent, 0.6)

    for intents in range(UserIntent):
        for itr in tqdm(range(iterations)):

            q = user.make_choice(intents)
            e = dbms.make_choice(q)

            #Add payoff 10 to the best intent-query pair matching
            if intents == e:
                user.update_qvalue(intents, q, 10)

        trials = 10
        cnt = 0
        for itr in range(trials):
            q = user.testing(intents)
            e = dbms.make_choice(q)
            if intents == e:
                cnt += 1
        accu = cnt / trials
        total += cnt
        print(accu)

    print(total)
    print(total/intents*trials)