# coding=utf-8
from algorithms.epsilon_greedy.standard import EpsilonGreedy
from algorithms.exp3.exp3 import Exp3
from algorithms.softmax.standard import Softmax
from algorithms.ucb.ucb1 import UCB1
from arms.bernoulli import BernoulliArm
from arms.normal import NormalArm
from testing_framework.tests import test_algorithm

execfile("core.py")

# 初始化三个臂
arm1 = BernoulliArm(0.7)  # 0.7的概率吐钱
arm1.draw()
arm1.draw()

arm2 = NormalArm(2.0, 0.5)
arm2.draw()
arm2.draw()

arm3 = BernoulliArm(0.2)  # 以0.2的概率吐钱
arm3.draw()
arm3.draw()

arms = [arm1, arm2, arm3]

n_arms = len(arms)

# 初始化四种算法
algo1 = EpsilonGreedy(0.1, [], [])
algo2 = Softmax(1.0, [], [])
algo3 = UCB1([], [])
algo4 = Exp3(0.2, [])

algos = [algo1, algo2, algo3, algo4]

for algo in algos:
    algo.initialize(n_arms)

for t in range(1000):  # 对于每一轮
    for algo in algos:  # 对于每一种算法
        chosen_arm = algo.select_arm()  # 选择一个臂
        reward = arms[chosen_arm].draw()  # 获得奖励
        algo.update(chosen_arm, reward)  # 根据获得的奖励更新值

print(algo1.counts)
print(algo1.values)

print(algo2.counts)
print(algo2.values)

print(algo3.counts)
print(algo3.values)

print(algo4.weights)

num_sims = 1000
horizon = 10
results = test_algorithm(algo1, arms, num_sims, horizon)
