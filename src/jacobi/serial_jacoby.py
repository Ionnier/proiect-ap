import sys
import numpy as np

def calc_roots_jacobi(A,b,t):
    n = len(A)
    x = np.zeros(n)

    for _ in range(t):
        x_new = np.ones(n)

        for i in range(n):
            first = -1/A[i][i]
            second = 0

            for j in range(n):
                if j != i:
                    second += np.dot(A[i][j],x[j])
    

            x_new[i] = first * (second - b[i])

        x = x_new
        print(x)

    return x

if len(sys.argv) != 3:
    exit(1)

CALE_FISIER_INPUT = sys.argv[1]
MAX_ITER = int(sys.argv[2])

data = np.genfromtxt(CALE_FISIER_INPUT)
A = data.transpose()[:-1].transpose()
b = data.transpose()[-1]
x_2 = calc_roots_jacobi(A,b,MAX_ITER)
print(x_2)