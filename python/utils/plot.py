# coding=utf-8
import matplotlib.pyplot as plt


def plot_lines(results, horizon, algorithm_types):
    """
    :param results: 多条曲线的纵坐标
    :param x: 横坐标
    :return: None
    """
    print(len(results))
    x = [i for i in range(horizon)]
    for result in results:
        plt.plot(x, result, linewidth=2.0, label=algorithm_types[results.index(result)])
    plt.xlabel("horizon")
    plt.ylabel("cumulative reward")
    plt.legend()
    plt.show()


def plot_task(task_name, algorithm_types, n_sim, horizon):
    global best_sim
    results = []
    for algo in algorithm_types:
        temp_cum_reward = 0
        file_name = task_name + "_" + algo
        f = open("./data/{f}.tsv".format(f=file_name), 'r')
        content = f.readlines()
        content = [line.split("\t") for line in content]

        # find best result
        for i in range(n_sim):
            current_cum_reward = float((content[i*horizon-1][4]))
            if current_cum_reward > temp_cum_reward:
                temp_cum_reward = current_cum_reward
                best_sim = i
        # record cum_reward in each step
        cum_reward = []
        for i in range(horizon):
            index = best_sim*horizon+i
            cum_reward.append(float(content[index][4]))
        results.append(cum_reward)

    plot_lines(results, horizon, algorithm_types)


if __name__ == '__main__':
    plot_task("task1", ["UCB1", "EpsilonGreedy"], 5, 250)