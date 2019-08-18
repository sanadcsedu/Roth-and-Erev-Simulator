import numpy as np


class FixedStrategy:

    def __init__(self, n, strategies, fixed_accuracy):
        self.n = n
        self.strategies = strategies
        self.original_mapping = np.zeros((n, strategies))
        self.fixed_accuracy = fixed_accuracy
        self.user_pd = np.zeros((n, strategies))
        self.dbms_pd = np.zeros((n, strategies))

        ch = []
        for i in range(strategies):
            ch.append(i)
        for i in range(n):
            mk_ch = np.random.choice(ch)
            ch.remove(mk_ch)
            self.original_mapping[i, mk_ch] = 1

        self.init_dbms_pd()

    def init_user_pd(self):
        return

    def init_dbms_pd(self):

        ch = []
        for i in range(self.strategies):
            ch.append(i)

        llen = int(self.n * self.fixed_accuracy)
        for i in range(self.n):
            if i < llen:
                self.dbms_pd[i] = self.original_mapping[i]
            else:
                posa = np.random.randint(0, self.strategies)
                posb = np.random.randint(0, self.strategies)
                p = np.random.randint(0, 10)
                # print(str(posa) + " " + str(posb) + " " + str(p))
                self.dbms_pd[i, posa] = p / 10
                self.dbms_pd[i, posb] = 1 - (p / 10)

    def make_choice(self, n):
        return np.argmax(self.dbms_pd[n])

    def testing(self, n):
        return np.argmax(self.dbms_pd[n])

    def show(self):
        print(self.original_mapping)
        print("\n\n")
        print(self.dbms_pd)


if __name__ == "__main__":
    fs = FixedStrategy(10, 10, 0.6)
    fs.show()
