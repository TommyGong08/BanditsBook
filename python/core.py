from enum import Enum
import matplotlib.pyplot as plt
import numpy as np


# Convenience functions
def ind_max(x):
    m = max(x)
    return x.index(m)


def set_algorithm(algorithm_type):
    if algorithm_type == "UCB1":
        return UCB1([], [])
    elif algorithm_type == "EpsilonGreedy":
        return EpsilonGreedy(0.1, [], [])
    elif algorithm_type == "ETC":
        return ETC(10000, [], [])
    elif algorithm_type == "EXP3":
        return Exp3(0.2, [])


# Need access to random numbers
import random

# Definitions of bandit arms
from arms.adversarial import *
from arms.bernoulli import *
from arms.normal import *

# Definitions of bandit algorithms
from algorithms.epsilon_greedy.standard import *
from algorithms.epsilon_greedy.annealing import *
from algorithms.softmax.standard import *
from algorithms.softmax.annealing import *
from algorithms.ucb.ucb1 import *
from algorithms.ucb.ucb2 import *
from algorithms.exp3.exp3 import *
from algorithms.hedge.hedge import *
from algorithms.etc.etc import *

# # Testing frameworks
from testing_framework.tests import *
from utils.file import *
from utils.plot import *
