
import numpy as np
import librosa
import random as rnd

from keras.utils import to_categorical

# TODO: Work with masked numpy arrays? Such that silence = mask

def symbol_to_hz(x, base_freq=440, base_key_estimator=np.average):
    base_key = int(base_key_estimator(x[np.nonzero(x)]))
    f = np.array(x, copy=True)
    f[np.nonzero(f)] = base_freq * 2**(((f[np.nonzero(f)]-base_key))/12)
    return f.astype(int)

def hz_to_pcp(x, octave_invariant=True):
    n = np.array([['' if item==0 else \
        librosa.hz_to_note(item, octave=not octave_invariant) \
            for item in row] for row in f])

    n_i = np.ndarray(n.shape, dtype=int)
    if octave_invariant:
        occ = ['', 'C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 
            'G', 'G♯', 'A', 'A♯', 'B'] 
    else:
        occ = [''] + np.array([ \
            [f'C{i}', f'C♯{i}', f'D{i}', f'D♯{i}', f'E{i}', f'F{i}', 
             f'F♯{i}', f'G{i}', f'G♯{i}', f'A{i}', f'A♯{i}', f'B{i}'] \
                for i in range(2,7)]).flatten().tolist()
    
    for (i, note) in enumerate(occ):
        n_i[np.where(n==note)] = i

    n_i = np.vstack((np.repeat(np.array(range(len(occ)))[...,None], 
                                4, axis=1), n_i))

    encoded = to_categorical(n_i[:,0])
    for v in range(1,x.shape[1]):
        encoded += to_categorical(n_i[:,v])

    encoded = encoded[len(occ):,...]

    encoded /= np.max(encoded)
    return occ, encoded

def symbol_to_pcp(x, octave_invariant=True, **kwargs):
    return hz_to_pcp(symbol_to_hz(x, **kwargs), octave_invariant)

def random_ohenc(l, n):
    rnd.seed()
    res = np.zeros(l)
    idcs = rnd.choices(list(range(l)), k=n)
    for idx in idcs:
        res[idx] += 1
    res /= n
    return res

def with_noise(pcp, n, n_v):
    return np.vstack((pcp, np.array([random_ohenc(pcp.shape[1], n_v) \
        for _ in range(n)]))) if n>0 else pcp

if __name__=='__main__':
    x = np.genfromtxt(r'C:\Users\Jesper\Documents\local study\1-2-ml\ml-semester-project\data\F.txt')
    f = symbol_to_hz(x)
    label_o, pcp_o = hz_to_pcp(f, False)
    label, pcp = hz_to_pcp(f, True)
    np.save('npys/F-pcp.npy', pcp)
    np.save('npys/F-pcp-o.npy', pcp_o)
    np.save('npys/F-pcp-lab.npy', label)
    np.save('npys/F-pcp-o-lab.npy', label_o)
    np.save('npys/F-hz.npy', f)