
def save_result(algorithm_type, results):
    f = open("demo1_{a}.tsv".format(a=algorithm_type), "w")
    for i in range(len(results[0])):
        f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")
    f.close()