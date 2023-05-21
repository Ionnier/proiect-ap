import numpy as np
import sys

ITERATION_LIMIT = 1000

if len(sys.argv) != 2:
    print('Invalid format!', file=sys.stderr)
    exit(1)
    
INPUT_FILE_PATH = sys.argv[1]
try:
    input_data = np.genfromtxt(INPUT_FILE_PATH)
    A = input_data.transpose()[:-1].transpose()
    b = input_data.transpose()[-1]
except:
    print('Input file reading failed!', file=sys.stderr)
    exit(1)

x = np.zeros_like(b)
for it_count in range(1, ITERATION_LIMIT):
    x_new = np.zeros_like(x)
    print(f"Iteration {it_count}: {x}")
    for i in range(A.shape[0]):
        s1 = np.dot(A[i, :i], x_new[:i])
        s2 = np.dot(A[i, i + 1 :], x[i + 1 :])
        x_new[i] = (b[i] - s1 - s2) / A[i, i]
    if np.allclose(x, x_new, rtol=1e-8):
        break
    x = x_new

print(f"Solution: {x}")
error = np.dot(A, x) - b
print(f"Error: {error}")
