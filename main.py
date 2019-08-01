import RothAndErevClass
import numpy as np
import ForRandomAgent

# rothanderev = RothAndErevClass.RothAndErevClass(5, 5, 0, 0, 0.0001, False)
# random_agent = ForRandomAgent.ForRandomAgent(10, 10, True)
# print(random_agent.make_choice())

experiments = iterations = 0
experiments = 5
iterations = 100
threshold = 0.8
cnt = 0

accuracy = np.zeros(10)

for exp in range(experiments):

    user = RothAndErevClass.RothAndErevClass(10, 10, 0, 0, 0.0001, False)
    dbms = RothAndErevClass.RothAndErevClass(10, 10, 0, 0, 0.0001, False)

    for intents in range(10):
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

        q = user.testing(intents)
        e = dbms.testing(q)
        if intents == e and intents == q:
            accuracy[intents] += 1
    print(np.sum(accuracy) / 10)
    accuracy.fill(0)

# print(accuracy)
# print(np.sum(accuracy) / 50)







