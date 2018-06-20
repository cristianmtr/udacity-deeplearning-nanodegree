import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.


def cross_entropy(Y, P):
    return -1 * sum(
        [
            (Y[i] * np.log(P[i]) + (1 - Y[i]) * np.log(1 - P[i]))
            for i in range(len(Y))
        ]
    )


if __name__ == "__main__":
    Y = [1, 0, 1, 1]
    P = [0.4, 0.6, 0.1, 0.5]
    print(cross_entropy(Y, P))
    print('should be: 4.8283137373')
