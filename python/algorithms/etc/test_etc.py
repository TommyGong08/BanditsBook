import core
import random

random.seed(1)
means = [0.1, 0.1, 0.9, 0.1, 0.1]
n_arms = len(means)
random.shuffle(means)
arms = map(lambda (mu): core.BernoulliArm(mu), means)
print("Best arm is " + str(core.ind_max(means)))

algo = core.ETC(5, [], [])
algo.initialize(n_arms)
results = core.test_algorithm(algo, arms, 5000, 250)

f = open("../../algorithms/etc/etc1_results.tsv", "w")

for i in range(len(results[0])):
    f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")

f.close()