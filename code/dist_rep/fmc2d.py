import scipy.fft
import matplotlib.pyplot as plt
import numpy as np
import os
import random as rnd
import math

from scipy.spatial.distance import cdist, pdist, squareform

path = os.path.dirname(__file__)

pcp_13 = np.load('%s/npys/F-pcp.npy' % path)
pcp_61 = np.load('%s/npys/F-pcp-o.npy' % path)

def set_pcp_kind(octave_invariant:bool):
    pcp = pcp_13 if octave_invariant else pcp_61

def pcp_to_fmc2d(x):
    fft2d = scipy.fft.fft2(x)
    r = np.real(fft2d)
    i = np.imag(fft2d)
    return scipy.fft.fftshift(np.sqrt(r*r + i*i))

def clean_fmc2d(x):
    fmc2d_fl = x.flatten()
    return fmc2d_fl[:fmc2d_fl.shape[0]//2 + 1]

def clean_fmc2d_2d(x):
    return x[...,:math.ceil(x.shape[1]/2)]

def cleaned_fmc2ds(x):
    return np.array([clean_fmc2d(pcp_to_fmc2d(patch)) for patch in x])

def cleaned_fmc2ds_2d(x):
    return np.array([clean_fmc2d_2d(pcp_to_fmc2d(patch)) for patch in x])

def pcp_to_fmc2ds(x, ignore_silence=True, window_size=20,
                 hop_size=None):
    hop_size = hop_size or window_size
    offset = x.shape[0] % (window_size - hop_size) 

    if ignore_silence:
        x = x[...,1:]

    indices = range(offset+window_size-1, x.shape[0])\
        [0 : x.shape[0]-offset : hop_size]

    pcp_windows = [x[(i-window_size):i,...] for i in indices]

    return indices, cleaned_fmc2ds(pcp_windows)

def pcp_to_fmc2ds_2d(x, ignore_silence=True, window_size=20,
                    hop_size=None):
    hop_size = hop_size or window_size
    offset = x.shape[0] % (window_size - hop_size) 

    if ignore_silence:
        x = x[...,1:]

    indices = range(offset+window_size-1, x.shape[0])\
        [0 : x.shape[0]-offset : hop_size]

    pcp_windows = [x[(i-window_size):i,...] for i in indices]

    return indices, cleaned_fmc2ds_2d(pcp_windows)

def distance_at(vec, idx, window=20, metric='euclidean', oct_inv=True):
    pcp = pcp_13 if oct_inv else pcp_61
    pcp1 = np.vstack((pcp[idx-window:idx,...], vec[None, ...]))
    pcp2 = pcp[idx-window-1:idx,...]
    return cdist(
        clean_fmc2d(pcp_to_fmc2d(pcp1))[None,...],
        clean_fmc2d(pcp_to_fmc2d(pcp2))[None,...],
        metric=metric
    )

def pcp_to_ssm(x, ignore_silence=True, window_size=20, 
               hop_size=None, metric='euclidean'):

    _, fmc2ds = pcp_to_fmc2ds(x, ignore_silence, window_size, hop_size)

    sdm = squareform(pdist(fmc2ds, metric=metric))
    sdm /= np.max(sdm)
    return 1 - sdm

def plot_ssm(ssm, figsize=(10,10), cmap='magma', interpolation='nearest',
             show=True, **kwargs):
    plt.figure(figsize=figsize)
    plt.matshow(ssm, cmap=cmap, interpolation=interpolation, **kwargs)
    if show:
        plt.show()


