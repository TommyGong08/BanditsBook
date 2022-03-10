import numpy as np
import random
execfile("core.py")


if __name__ == '__main__':
    task_name = "task1"

    horizon = 1000
    n_arms = 10
    best_arms_number = random.randint(0, n_arms)
    print("the best arm is : No. {a}".format(a=best_arms_number))

    means = 10 * [0.1]
    means[best_arms_number - 1] = 0.9
    arms = map(lambda (mu): BernoulliArm(mu), means)

    # set algorithms
    algorithm_types = ["UCB1", "EpsilonGreedy"]
    for algo in algorithm_types:
        algorithm_type = algo
        algo = set_algorithm(algorithm_type)
        algo.initialize(n_arms)
        results = test_algorithm(algo, arms, 5, horizon)

        # save result
        save_result(task_name, algorithm_type, results)
        print("#####")
    # plot
    plot_task(task_name, algorithm_types, 5, horizon)
