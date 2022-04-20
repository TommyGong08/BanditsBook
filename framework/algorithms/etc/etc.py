# coding=utf-8
import random


def ind_max(x):
    m = max(x)
    return x.index(m)


class ETC():
    def __init__(self, m, counts, values):
        self.m = m  # explore阶段，每个臂要执行的次数
        self.counts = counts
        self.values = values
        return

    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]  # 每种算法在每个臂上的执行次数， 数组长度为臂的数量
        self.values = [0.0 for col in range(n_arms)]  # 每种算法在每个臂上获得的奖励
        return

    def select_arm(self):
        n_arms = len(self.counts)  # 臂的数量K
        total_counts = sum(self.counts)  # 当前执行的轮数t
        if total_counts > n_arms * self.m:
            return ind_max(self.values)
        else:  # t mod k +1
            return total_counts % n_arms

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
