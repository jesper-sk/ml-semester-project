import math
import numpy as np
from scipy.special import softmax

BASE_KEY = 54  # Key that denotes the base frequency
BASE_FREQ = 440  # Hz

CHROMA = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
CHROMA_R = 1

C5 = [1, 8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6]
C5_R = 1


def note_to_vector(x, offset, total):
    if isinstance(x, np.ndarray):
        return np.hstack(tuple([note_to_vector(xi, offset, total) for xi in x]))
    elif x == 0:
        return np.repeat(0., 5)
    else:
        chroma_x, chroma_y = note_to_chroma_xy(x)
        c5_x, c5_y = note_to_c5_xy(x)
        pitch = note_to_log_pitch(x, offset, total)

        return np.array([pitch, chroma_x, chroma_y, c5_x, c5_y])


def note_to_chroma_xy(note):
    n = (note-BASE_KEY) % 12
    chroma_rad = n * (math.pi/6)
    x = CHROMA_R * math.sin(chroma_rad)
    y = CHROMA_R * math.cos(chroma_rad)
    return (x, y)


def note_to_c5_xy(note):
    n = (note-BASE_KEY) % 12
    c5_rad = (C5[n] - 1) * (math.pi/6)
    x = C5_R * math.sin(c5_rad)
    y = C5_R * math.cos(c5_rad)
    return (x, y)


def note_to_log_pitch(note, offset, total):
    min_p = 2 * math.log2(2**((offset - BASE_KEY)/12) * BASE_FREQ)
    max_p = 2 * math.log2(2**((offset + total - 1 - BASE_KEY)/12) * BASE_FREQ)
    pitch = 2 * math.log2((2**((note - BASE_KEY)/12) * BASE_FREQ)) - max_p\
        + ((max_p - min_p)/2)
    return pitch


def encode_note_duration(F, voice=0):
    single = F[..., voice]

    prev = single[0]
    dur = 0

    durs = []
    notes = [single[0]]

    for note in single:
        if note == prev:
            dur += 1
        else:
            notes.append(note)
            durs.append(dur)
            prev = note
            dur = 1

    durs.append(dur)

    durs = np.array(durs)
    notes = np.array(notes)

    return notes, durs


def encode_duration(F, voice=None):
    root = F if voice is None else F[..., voice]

    prev = root[0]
    dur = 0

    durs = []
    items = [root[0]]

    for item in root:
        if np.all(item==prev):
            dur += 1
        else:
            items.append(item)
            durs.append(dur)
            prev = item
            dur = 1
    
    durs.append(dur)

    durs = np.array(durs)
    items = np.array(items)

    return items, durs


def biased(X):
    return np.hstack(
        (np.ones((len(X), 1)), X)
    )


def windowed(X, window_size=10, hop_size=1):
    hop_size = hop_size or window_size
    offset = X.shape[0] % (window_size - hop_size)

    indices = (range(offset+window_size-1, X.shape[0])
               [0:X.shape[0]-offset:hop_size])

    X_windows = np.array([X[(i-window_size):i, ...].flatten()
                         for i in indices])

    return (X_windows, indices)


if __name__ == "__main__":
    F = np.genfromtxt(r"C:\Users\Jesper\Documents\local study\1-2-ml\ml-"
                      + r"semester-project\data\F.txt", dtype=int)
    pass
