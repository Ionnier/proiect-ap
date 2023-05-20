from mpi4py import MPI
import sys
import numpy
import sympy
import time

MASTER = 0

if len(sys.argv) != 2:
    exit(1)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

CALE_FISIER_INPUT = sys.argv[1]

data = None
indexes = None

starttimestamp = int(round(time.time() * 1000))
if rank == MASTER:
    data = numpy.genfromtxt(CALE_FISIER_INPUT)
    indexes = []
    for i in range(len(data)):
        for j in range(len(data[0])):
            indexes.append((i, j))
    indexes = numpy.array_split(indexes, size)


indexes = comm.scatter(indexes, root=MASTER)
data = comm.bcast(data, MASTER)

A = data
C = [A]

for i in range(len(A)-1):
    calculatedIndexes = []
    for index in indexes:
        j = index[0]
        k = index[1]
        calculatedIndexes.append(
            (
                tuple(index),
                C[i][j][k] -
                (C[i][j][i+1])/(C[i][i+1][i+1]) * (C[i][i+1][k])
            )
        )
    gathered_data = comm.gather(calculatedIndexes, root=MASTER)

    newC = None
    if rank == MASTER:
        newC = numpy.copy(data)
        for calculatedIndex in gathered_data:
            for hey in calculatedIndex:
                index = hey[0]
                value = hey[1]
                newC[index[0]][index[1]] = value
    newC = comm.bcast(newC, MASTER)
    C.append(newC)


if rank == MASTER:
    x = []
    knowns = []
    for i in range(len(A)):
        x.append(f"x{i}")
        knowns.append(None)
    variables = sympy.symbols(" ".join(x))
    print(f"Duration: {int(round(time.time() * 1000)) - starttimestamp}")
    for i in range(len(C)-1, -1, -1):
        for equation in C[i]:
            hey = 0
            for s in range(len(equation)-1):
                if (knowns[s] == None):
                    hey += variables[s] * equation[s]
                else:
                    hey += knowns[s] * equation[s]
            equation = sympy.Eq(hey, equation[-1])
            solutions = sympy.solve(equation, variables)
            for s in solutions:
                flag = False
                for idx in range(len(s)):
                    variable = s[idx]
                    if (type(variable) != sympy.core.symbol.Symbol and knowns[idx] == None):
                        knowns[idx] = variable
                        flag = True
                        break
                if flag:
                    continue
    print(knowns)
