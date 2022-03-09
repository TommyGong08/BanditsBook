# coding=utf-8
import random


def ind_max(x):
    m = max(x)
    return x.index(m)


class EpsilonGreedy():
    def __init__(self, epsilon, counts, values):
        self.epsilon = epsilon
        self.counts = counts
        self.values = values
        return

    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]  # 每种算法在每个臂上的执行次数
        self.values = [0.0 for col in range(n_arms)]  # 每种算法在每个臂上获得的奖励
        return

    def select_arm(self):  # 以1-epsilon的概率选择最优的动作
        if random.random() > self.epsilon:
            return ind_max(self.values)
        else:  # 以ε的概率随机选择
            return random.randrange(len(self.values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return
