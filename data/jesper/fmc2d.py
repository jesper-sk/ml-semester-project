from data.jesper.transform import BASE_KEY
import numpy as np
import scipy.fft
import librosa
from keras.utils import to_categorical

# TODO: Work with masked numpy arrays? Such that silence = mask

def pcp_to_fmc2d(X):
    fft2d = scipy.fft.fft2(X)
    r = np.real(fft2d)
    i = np.imag(fft2d)
    return np.sqrt(r*r + i*i)

def clean_fmc2d(X):
    fmc2d_fl = scipy.fft.fftshift(X).flatten()
    return fmc2d_fl[:fmc2d_fl.shape[0]//2 + 1]

def symbol_to_hz(x, base_freq=440, base_key_estimator=np.average):
    base_key = int(base_key_estimator(x[np.nonzero(x)]))
    f = np.array(x, copy=True)
    f[np.nonzero(f)] -= base_key
    f[np.nonzero(f)] = base_freq * 2**(f[np.nonzero(f)]/12)
    return f.astype(int)

def hz_to_pcp(x, octave=True, window_size=1):
    n = np.array([['' if item==0 else librosa.hz_to_note(item, octave=octave) for item in row] for row in f])
    n_i = np.ndarray(n.shape, dtype=int)
    occ = ['', 'C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'] if not octave else \
        [''] + np.array([[f'C{i}', f'C♯{i}', f'D{i}', f'D♯{i}', f'E{i}', f'F{i}', f'F♯{i}', f'G{i}', f'G♯{i}', f'A{i}', f'A♯{i}', f'B{i}'] for i in range(2,7)]).flatten().tolist()
    
    for (i, note) in enumerate(occ):
        n_i[np.where(n==note)] = i

    encoded = to_categorical(n_i[:,0])
    for v in range(1,x.shape[1]):
        encoded += to_categorical(n_i[:,v])
