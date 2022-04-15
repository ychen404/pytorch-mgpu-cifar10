# hellinger_distance_demo.py

import numpy as np

def H(p, q):
    # distance between p an d
    # p and q are np array probability distributions
    n = len(p)
    sum = 0.0
    for i in range(n):
        sum += (np.sqrt(p[i]) - np.sqrt(q[i]))**2
    result = (1.0 / np.sqrt(2.0)) * np.sqrt(sum)
    return result


def main():
    print("\nBegin Hellinger distance from scratch demo ")
    np.set_printoptions(precision=4, suppress=True)

    p = np.array([2.0, 10.0, 0.0, 0.0], dtype=np.float32)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    print("\nThe P distribution is: ")
    print(p)
    print("\nThe Q distribution is: ")
    print(q)

    h_pq = H(p, q)
    h_qp = H(q, p)

    print("\nH(P,Q) = %0.6f " % h_pq)
    print("H(Q,P) = %0.6f " % h_qp)

    print("\nEnd demo ")


if __name__ == "__main__":
    main()