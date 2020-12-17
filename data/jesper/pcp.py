#from data.jesper.transform import BASE_KEY
import numpy as np

import librosa
import matplotlib.pyplot as plt
from keras.utils import to_categorical

# TODO: Work with masked numpy arrays? Such that silence = mask

def symbol_to_hz(x, base_freq=440, base_key_estimator=np.average):
    base_key = int(base_key_estimator(x[np.nonzero(x)]))
    f = np.array(x, copy=True)
    f[np.nonzero(f)] -= base_key
    f[np.nonzero(f)] = base_freq * 2**(f[np.nonzero(f)]/12)
    return f.astype(int)

def hz_to_pcp(x, octave=True):
    n = np.array([['' if item==0 else librosa.hz_to_note(item, octave=octave) for item in row] for row in f])
    n_i = np.ndarray(n.shape, dtype=int)
    occ = ['', 'C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'] if not octave else \
        [''] + np.array([[f'C{i}', f'C♯{i}', f'D{i}', f'D♯{i}', f'E{i}', f'F{i}', f'F♯{i}', f'G{i}', f'G♯{i}', f'A{i}', f'A♯{i}', f'B{i}'] for i in range(2,7)]).flatten().tolist()
    
    for (i, note) in enumerate(occ):
        n_i[np.where(n==note)] = i

    n_i = np.vstack((np.repeat(np.array(range(len(occ)))[...,None], 4,axis=1), n_i))

    encoded = to_categorical(n_i[:,0])
    for v in range(1,x.shape[1]):
        encoded += to_categorical(n_i[:,v])

    encoded = encoded[len(occ):,...]

    encoded /= np.max(encoded)
    return occ, encoded

if __name__=='__main__':
    x = np.genfromtxt(r'C:\Users\Jesper\Documents\local study\1-2-ml\ml-semester-project\data\F.txt')
    print(x[40:50,...])
    f = symbol_to_hz(x)
    print(f[40:50,...])
    label_o, pcp_o = hz_to_pcp(f, False)
    label, pcp = hz_to_pcp(f, True)
    np.save('../npys/F-pcp.npy', pcp_o)
    np.save('../npys/F-pcp-o.npy', pcp_o)
    np.save('../npys/F-pcp-lab.npy', label)
    np.save('../npys/F-pcp-o-lab.npy', label_o)
    np.save('../npys/F-hz.npy', f)