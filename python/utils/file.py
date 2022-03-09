
def save_result(task_name, algorithm_type, results):
    f = open("data/{T}_{a}.tsv".format(T=task_name, a=algorithm_type), "w")
    for i in range(len(results[0])):
        f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")
    f.close()