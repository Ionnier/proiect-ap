import numpy
import sys

if (len(sys.argv) != 2):
    exit(1)

CALE_FISIER_INPUT = sys.argv[1]
A = numpy.genfromtxt(CALE_FISIER_INPUT)

C = [A]

for i in range(len(A)-1):
    newC = numpy.copy(A)
    for j in range(len(A)):
        for k in range(len(A[0])):
            newC[j][k] = C[i][j][k] - \
                (C[i][j][i+1])/(C[i][i+1][i+1]) * (C[i][i+1][k])
    C.append(newC)

print(C)
