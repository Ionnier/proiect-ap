import sys
import time
import numpy as np
from mpi4py import MPI

if len(sys.argv) != 3:
    print('Invalid format!', file=sys.stderr)
    exit(1)

MASTER_PROCESS = 0
MAX_NUM_OF_ITERATIONS = int(sys.argv[2])
CURRENT_ITERATION = 0
TOLERANCE = 1e-8

def compute_x(index, A, b, prev_x, curr_x):
    n = len(A)
    first = -1 / A[index]
    second = 0
    third = 0

    if index == 0:
        for i in range(1, n):
            third += np.dot(A[i], prev_x[i])
    else:
        for i in range(0, index):
            second += np.dot(A[i], curr_x[i])

        for i in range(index + 1, n):
            third += np.dot(A[i], prev_x[i])

    local_x = first * (second + third - b)
    return local_x


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

input_data = None
A = None
b = None

def getTimeStamp():
    return int(round(time.time() * 1000))


prevTimestamp = None


def log(message, force=False):
    global prevTimestamp
    if (rank == MASTER_PROCESS or force):
        currTimestamp = getTimeStamp()
        fin = ""
        if (prevTimestamp != None):
            fin = f"{currTimestamp - prevTimestamp}"
        print(f"{message} timestamp = {currTimestamp} diff = {fin}")
        prevTimestamp = currTimestamp

starttimestamp = int(round(time.time() * 1000))
if rank == MASTER_PROCESS:
    log(f"Start read from file")
    INPUT_FILE_PATH = sys.argv[1]
    try:
        input_data = np.genfromtxt(INPUT_FILE_PATH)
        A = input_data.transpose()[:-1].transpose()
        b = input_data.transpose()[-1]
    except:
        print('Input file reading failed!', file=sys.stderr)
        exit(1)

local_A = []
local_b = []
if rank == MASTER_PROCESS:
    matrix_size = A.shape[0]

    if matrix_size % nprocs == 0:
        SPLIT_FACTOR = int(matrix_size / nprocs)
    else:
        SPLIT_FACTOR = int(matrix_size / nprocs) + 1

    # slave
    for i in range(1, nprocs):
        local_A = []
        local_b = []
        for j in range(i * SPLIT_FACTOR, (i + 1) * SPLIT_FACTOR):
            if j >= matrix_size:
                break
            local_A.append(A[j])
            local_b.append(b[j])
        comm.send((local_A, local_b, SPLIT_FACTOR), i)

    # master
    local_A = []
    local_b = []
    for j in range(0, SPLIT_FACTOR):
        local_A.append(A[j])
        local_b.append(b[j])
else:
    local_A, local_b, SPLIT_FACTOR = comm.recv(source=MASTER_PROCESS)

initial_x = None
prev_x = []
curr_x = []
local_x = None
solution = []

if rank == MASTER_PROCESS:
    prev_x = np.ones(local_A[0].shape[0])
    initial_x = prev_x

log(f"Start iterations")
while CURRENT_ITERATION < MAX_NUM_OF_ITERATIONS:
    if rank != MASTER_PROCESS:
        prev_x, curr_x = comm.recv(source=rank - 1)
    elif rank == MASTER_PROCESS and CURRENT_ITERATION != 0:
        prev_x, curr_x, solution = comm.recv(source=nprocs - 1)

        if np.linalg.norm(solution - initial_x) < TOLERANCE:
            break

    for matrix_line in range(len(local_A)):
        local_x = compute_x(rank * SPLIT_FACTOR + matrix_line,
                            local_A[matrix_line], local_b[matrix_line], prev_x, curr_x)
        curr_x.append(local_x)

    if rank != nprocs - 1:
        comm.send((prev_x, curr_x), rank + 1)
    else:
        solution = curr_x
        # print('Iteration:', CURRENT_ITERATION, '- x:', curr_x)
        comm.send((curr_x, [], solution), MASTER_PROCESS)

    CURRENT_ITERATION += 1
log(f"Finish iterations")

if rank == MASTER_PROCESS:
    print('SOLUTION', solution)
    log(f"Print solution")
