import copy
import sys
import time
from mpi4py import MPI
import numpy as np

MASTER = 0

if len(sys.argv) != 3:
    exit(1)

CALE_FISIER_INPUT = sys.argv[1]
MAX_ITER = int(sys.argv[2])
data = None

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

A = None
b = None

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
    data = np.genfromtxt(CALE_FISIER_INPUT)
    A = data.transpose()[:-1].transpose()
    b = data.transpose()[-1]
    for i in range(len(A)):
        sum = 0
        for j in range(len(A[i])):
            if j != i:
                sum += abs(A[i][j])
        if abs(A[i][i]) <= sum:
            print('Matricea nu este diagonal dominanta!')
            break


local_A = []
local_b = []


if rank == MASTER:
    all_indexes = []
    indexes = range((rank*len(A))//size, ((rank+1) * len(A))//size)
    if size == 1:
        local_A = A
        local_b = b
    else:
        for i in indexes:
            local_A.append(A[i])
            local_b.append(b[i])
        for i in range(1, size):
            all_indexes.append(
                list(range((i*len(A))//size, ((i+1) * len(A))//size)))
        # print(all_indexes)
        for idx in range(len(all_indexes)):
            data_A = []
            data_b = []
            for i in all_indexes[idx]:
                data_A.append(A[i])
                data_b.append(b[i])
            log(f"Start sending")
            comm.send((data_A, data_b), 1+idx)
else:
    log(f"Start receiving")
    local_A, local_b = comm.recv(source=MASTER)

n = len(local_A[0])
indexes = range((rank*n)//size, ((rank+1) * n)//size)
x_new = np.zeros(n)
log(f"Start iterations")
for _ in range(MAX_ITER):
    # print(x_new)
    x_new_copy = copy.deepcopy(x_new)
    for index in indexes:
        # print(local_A,local_A[indexes.index(index)], index)
        first = -1 / local_A[indexes.index(index)][index]
        second = 0
        for j in range(n):
            if j != index:
                second += np.dot(local_A[indexes.index(index)]
                                 [j], x_new_copy[j])

        x_new[index] = first * (second - local_b[indexes.index(index)])

    log(f"Start gather")
    x_new2 = comm.allgather((x_new, list(indexes)))
    # print(x_new2)
    for el in x_new2:
        for idx in el[1]:
            x_new[idx] = el[0][idx]
log(f"Finish iterations")
if rank == MASTER:
    print(x_new)


# if rank == 0:
#     print("Radacinile sunt:", x)
