import argparse
import numpy as np
import dist_rep as dr
from dist_rep import pcp, fmc2d
import transform as t

import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--paths', nargs='+', type=str, required=True)
    parser.add_argument('-f', type=str, required=True)
    parser.add_argument('-o', type=int, required=False)

    args = parser.parse_args()
    
    total = []

    f = np.genfromtxt(args.f, dtype=int)

    for path in args.paths:

        file = np.genfromtxt(path, dtype=int, delimiter=',')
        samples = []
        for row in file:
            samples = np.append(samples, np.repeat(row[0], row[1]))

        total.append(samples)

    ln = min([len(x) for x in total])

    song = total[0][:ln].reshape(-1,1)
    if len(total) > 1:
        for voice in total[1:]:
            song = np.hstack((song,voice[:ln].reshape(-1,1)))

    f = f[args.o:,:len(total)]
    print(f.shape)

    cont = np.vstack((f, song))

    offset = np.min(cont[cont!=0])
    total = np.max(cont)

    occ, pcp = pcp.symbol_to_pcp(cont, False)
    ssm = fmc2d.pcp_to_ssm(pcp, window_size=20, hop_size=1)

    plt.matshow(ssm, cmap='magma')
    plt.show()

    