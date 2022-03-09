# coding=utf-8
import random


class NormalArm():
    def __init__(self, mu, sigma):
        self.mu = mu  # 均值
        self.sigma = sigma  # 方差

    def draw(self):
        return random.gauss(self.mu, self.sigma)
