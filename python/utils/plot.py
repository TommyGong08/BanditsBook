# coding=utf-8
import matplotlib.pyplot as plt


def plot_lines(results, horizon, algorithm_types):
    """
    :param results: 多条曲线的纵坐标
    :param x: 横坐标
    :return: None
    """
    plt.subplot(1, 2, 1)
    x = [i for i in range(horizon)]
    for result in results:
        plt.plot(x, result, linewidth=2.0, label=algorithm_types[results.index(result)])
    plt.xlabel("horizon")
    plt.ylabel("cumulative reward")
    plt.legend()


def plot_regret(results, horizon, algorithm_types):
    """
        :param results: 多条曲线的纵坐标
        :param x: 横坐标
        :return: None
        """
    plt.subplot(1, 2, 2)
    x = [i for i in range(horizon)]
    for result in results:
        plt.plot(x, result, linewidth=2.0, label=algorithm_types[results.index(result)])
    plt.xlabel("horizon")
    plt.ylabel("regrets")
    plt.legend()
    plt.show()


def plot_task(task_name, algorithm_types, n_sim, horizon):
    global best_sim
    results = []
    max_rewards = []
    for algo in algorithm_types:
        temp_cum_reward = 0
        file_name = task_name + "_" + algo
        f = open("../data/{f}.tsv".format(f=file_name), 'r')
        content = f.readlines()
        content = [line.split("\t") for line in content]

        # 找到最好的结果
        for i in range(n_sim):
            current_cum_reward = float((content[i*horizon-1][4]))
            if current_cum_reward > temp_cum_reward:
                temp_cum_reward = current_cum_reward
                best_sim = i
        # record cum_reward in each step
        cum_reward = []
        best_reward = []
        for i in range(horizon):
            index = best_sim*horizon+i
            cum_reward.append(float(content[index][4]))
            best_reward.append(float(content[index][1]) - float(content[index][4]))
        results.append(cum_reward)
        max_rewards.append(best_reward)

    plot_lines(results, horizon, algorithm_types)
    plot_regret(max_rewards, horizon, algorithm_types)
    plt.show()


if __name__ == '__main__':
    plot_task("task_delta_0.4-0.7_10arms_Bernoulli_1000000_ETCm1000", ["UCB1", "EpsilonGreedy", "ETC"], 5, 100000)