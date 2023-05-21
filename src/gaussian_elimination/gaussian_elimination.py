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


def getTimeStamp():
    return int(round(time.time() * 1000))


prevTimestamp = None


def log(message, force=False):
    global prevTimestamp
    if (rank == MASTER or force):
        currTimestamp = getTimeStamp()
        fin = ""
        if (prevTimestamp != None):
            fin = f"{currTimestamp - prevTimestamp}"
        print(f"{message} timestamp = {currTimestamp} diff = {fin}")
        prevTimestamp = currTimestamp


starttimestamp = int(round(time.time() * 1000))
if rank == MASTER:
    log(f"Start read from file")
    data = numpy.genfromtxt(CALE_FISIER_INPUT)
    indexes = []
    for i in range(len(data)):
        for j in range(len(data[0])):
            indexes.append((i, j))
    indexes = numpy.array_split(indexes, size)

log(f"Start scatter")
indexes = comm.scatter(indexes, root=MASTER)
log(f"Start bcast")
data = comm.bcast(data, MASTER)

A = data
C = [A]

log(f"Start iterations")
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
log(f"Finish iterations")


if rank == MASTER:
    x = []
    knowns = []
    for i in range(len(A)):
        x.append(f"x{i}")
        knowns.append(None)
    log(f"Start substitution")
    for i in range(len(C)-1, -1, -1):
        for equation in C[i]:
            hey = 0
            curr_x = -1
            curr_v = -1
            for s in range(len(equation)-1):
                if (knowns[s] == None and equation[s] != 0):
                    if (curr_x != -1):
                        print("Somewhere, something went wrong")
                        exit(0)
                    curr_x = s
                    curr_v = equation[s]
                else:
                    if (knowns[s] != None):
                        hey += knowns[s] * equation[s]
            total_value = equation[-1] - hey
            knowns[curr_x] = total_value / curr_v
            break
    log(f"Finish substitution")
    print(knowns)
