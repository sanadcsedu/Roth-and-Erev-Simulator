import UCB1
import FixedStrategy
import numpy as np
from tqdm import tqdm

experiments = iterations = 0
experiments = 1
iterations = 100
threshold = 0.8
cnt = 0
UserIntent = 10
QueryPerIntent = 10

accuracy = np.zeros(10)

pltx = []
plty = []
total = 0
for exp in range(experiments):

    user = UCB1.UCB1(UserIntent, QueryPerIntent)
    dbms = FixedStrategy.FixedStrategy(QueryPerIntent, UserIntent, 0.6)

    #For Printing DBMS Query-to-Intent Mapping
    # for query in range(QueryPerIntent):
    #     _intent = dbms.make_choice(query)
    #     print("Query " + str(query) + " intent " + str(_intent))

    for intents in range(UserIntent):
        for itr in tqdm(range(iterations)):

            q = user.make_choice(intents)
            e = dbms.make_choice(q)

            #Add payoff 10 to the best intent-query pair matching
            if intents == e:
                user.update_qvalue(intents, q, 10)
            else:
                user.update_qvalue(intents, q, 1)

        trials = 1
        cnt = 0
        for itr in range(trials):
            q = user.testing(intents)
            e = dbms.make_choice(q)
            print("%d %d" % (q, e))
            if intents == e:
                cnt += 1
        accu = cnt / trials
        total += cnt
        print(accu)

    print(total)
    print(total/UserIntent*trials)