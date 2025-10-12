# weavehacks2/utils/bandit.py
import random

class EpsGreedyBandit:
    def __init__(self, arms, eps=0.25):
        self.arms = list(arms)          # list of rulebook dicts
        self.name = [a["name"] for a in arms]
        self.eps = eps
        self.n = [0]*len(arms)
        self.value = [0.0]*len(arms)    # estimated cost (lower is better)

    def select(self):
        # explore
        if random.random() < self.eps:
            i = random.randrange(len(self.arms))
            return i, self.arms[i]
        # exploit best so far (if all 0 counts, pick random)
        best = None
        best_i = 0
        for j in range(len(self.arms)):
            v = self.value[j] if self.n[j] > 0 else float("inf")
            if best is None or v < best:
                best, best_i = v, j
        if self.n[best_i] == 0:
            best_i = random.randrange(len(self.arms))
        return best_i, self.arms[best_i]

    def update(self, i, cost):
        self.n[i] += 1
        alpha = 1.0 / self.n[i]
        self.value[i] = (1 - alpha) * self.value[i] + alpha * float(cost)
