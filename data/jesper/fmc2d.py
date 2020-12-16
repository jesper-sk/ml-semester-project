from data.jesper.transform import BASE_KEY
import numpy as np
import scipy.fft

def pcp_to_fmc2d(X):
    fft2d = scipy.fft.fft2(X)
    r = np.real(fft2d)
    i = np.imag(fft2d)
    return np.sqrt(r*r + i*i)

def clean_fmc2d(X):
    fmc2d_fl = scipy.fft.fftshift(X).flatten()
    return fmc2d_fl[:fmc2d_fl.shape[0]//2 + 1]

def symbol_to_freq(X, base_freq=440, base_key_estimator=np.average):
    base_key = int(base_key_estimator(X[np.nonzero(X)]))
    X -= base_key
    X[X == -base_key] = 0
    X = base_freq * 2**(X/12)

