# coding=utf-8
import numpy as np
import random
import core


if __name__ == '__main__':
    task_name = "task_delta_0.4-0.5_10arms_Normal_1000000_ETCm10000"

    horizon = 1000000  # 执行的轮数
    n_arms = 10  # 臂的数量
    best_arms_number = random.randint(0, n_arms)
    print("the best arm is : No. {a}".format(a=best_arms_number))

    means = 10 * [0.4]
    means[best_arms_number - 1] = 0.5
    # arms = map(lambda (mu): core.BernoulliArm(mu), means)  # Bernoulli Arm
    arms = map(lambda (mu): core.NormalArm(mu, sigma=0.1), means)

    # set algorithms
    algorithm_types = ["UCB1", "EpsilonGreedy", "ETC"]
    for algo in algorithm_types:
        # 对于每种算法执行一次任务
        algorithm_type = algo
        algo = core.set_algorithm(algorithm_type)  # core中返回相应算法的结构体
        algo.initialize(n_arms)
        results = core.test_algorithm(algo, arms, 5, horizon)

        # save result
        core.save_result(task_name, algorithm_type, results)

    # plot
    core.plot_task(task_name, algorithm_types, 5, horizon)
