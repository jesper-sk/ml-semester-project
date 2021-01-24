import numpy as np

f = np.genfromtxt(r'..\..\data\F.txt')

# c_unique = np.array([0,0,0,0])
# c_same = np.array([0,0,0,0])
# prev = [None, None, None, None]
# for row in f:
#     eq = [p==f for (p,f) in zip(prev, row)]
#     diff = not eq
#     c_unique = [count+this for (count, this) in zip(c_unique, eq)]
#     c_same = [0 if not this else count+this for (count, this) in zip(c_same, eq)]

#     uneven = c_same % 2

#     prev = row

prev = [0,0,0,0]
counts = [0,0,0,0]
for r_ind, row in enumerate(f):
    for c_ind, col in enumerate(row):
        if (col != prev[c_ind]):
            if (counts[c_ind] % 2 != 0):
                print(f'Found uneven amount of values in row {r_ind}, column {c_ind}, count {counts[c_ind]}')
            counts[c_ind] = 1
        else:
            counts[c_ind] += 1
    prev = row