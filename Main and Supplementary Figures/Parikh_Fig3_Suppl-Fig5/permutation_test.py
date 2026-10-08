import random
import numpy as np

def random_permutation(data_length, pop_n):
    group1 = random.sample(range(data_length), pop_n)
    group2 = list(set(range(data_length)) - set(group1))
    return [group1, group2]

def data_processing(test_population):
    return np.mean(test_population, axis = 0).max()

def permutation_test(data, pop_n, pop_thr, num_runs, process):
    s = 0
    seen = set()
    hist = []
    for i in range(num_runs):
        sen = 0
        while True:
            indices = random_permutation(len(data), pop_n)
            key = frozenset(indices[0])
            if key not in seen:
                seen.add(key)
                break
            elif sen > 2:
                break
            else:
                sen += 1
                print('hash collision')
                print(sen)

        test_population = [data[i] for i in indices[0]]
        T = process(test_population)
        hist.append(T)
        if T >= pop_thr:
            s += 1
    return s / num_runs, hist

