import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.


def softmax(L):
    denominator = 0
    for item in L:
        denominator += np.e**item
    results = []
    for item in L:
        prob_item = np.e**item / denominator
        results.append(prob_item)

    return results

if __name__ == "__main__":
    softmax([5, 6, 7])
    # [0.09003057317038046, 0.24472847105479764, 0.6652409557748219]
