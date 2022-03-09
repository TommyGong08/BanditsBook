import numpy as np
import random
execfile("core.py")


if __name__ == '__main__':
    n_arms = 10
    best_arms_number = random.randint(0, n_arms)
    print("the best arm is : No. {a}".format(a=best_arms_number))

    means = 10 * [0.1]
    means[best_arms_number - 1] = 0.9
    arms = map(lambda (mu): BernoulliArm(mu), means)

    # set algorithm type
    algorithm_type = "UCB1"
    algo = set_algorithm(algorithm_type)
    algo.initialize(n_arms)
    results, best_sim = test_algorithm(algo, arms, 5, 250)
    print("best_sim is {a}".format(a=best_sim))

    # save result
    save_result(algorithm_type, results)

    # set e-greedy
    algorithm_type = "EpsilonGreedy"
    algo = set_algorithm(algorithm_type)
    algo.initialize(n_arms)
    results, best_sim = test_algorithm(algo, arms, 5, 250)

    # save result
    save_result(algorithm_type, results)


