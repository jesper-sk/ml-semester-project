import scipy.fft
import numpy as np

def pcp_to_fmc2d(X):
    fft2d = scipy.fft.fft2(X)
    r = np.real(fft2d)
    i = np.imag(fft2d)
    return np.sqrt(r*r + i*i)

def clean_fmc2d(X):
    fmc2d_fl = scipy.fft.fftshift(X).flatten()
    return fmc2d_fl[:fmc2d_fl.shape[0]//2 + 1]