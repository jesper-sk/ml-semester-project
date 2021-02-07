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
        min_note = offset
        max_note = offset + total - 1

        note = (x-BASE_KEY) % 12

        chroma_rad = (CHROMA[note] - 1) * (math.pi/6)  # 2pi / 12
        c5_rad = (C5[note] - 1) * (math.pi/6)

        chroma_x = CHROMA_R * math.sin(chroma_rad)
        chroma_y = CHROMA_R * math.cos(chroma_rad)

        c5_x = C5_R * math.sin(c5_rad)
        c5_y = C5_R * math.cos(c5_rad)

        n = x - BASE_KEY
        freq = 2**(n/12) * BASE_FREQ

        min_p = 2 * math.log2(2**((min_note - BASE_KEY)/12) * BASE_FREQ)
        max_p = 2 * math.log2(2**((max_note - BASE_KEY)/12) * BASE_FREQ)

        pitch = 2 * math.log2(freq) - max_p + ((max_p - min_p)/2)

        return np.array([pitch, chroma_x, chroma_y, c5_x, c5_y])


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

# def construct_feature(note, duration, fprops):
#     offset, total = fprops
#     vector = note_to_vector(note, offset, total)
#     return np.hstack((duration, vector))

# def construct_features(notes, durations, fprops, inference=False):
#     offset, total = fprops

#     vecs = np.array([note_to_vector(note, offset, total) for note in notes])
#     if inference:
#         print('shape', vecs.shape, 'vecs', vecs)

#     # Normalize durations between -1, 1
#     durations = durations - durations.mean()
#     if np.abs(durations).max() == 0:
#         durations = np.zeros(len(durations))
#     else:
#         durations = durations / np.abs(durations).max()

#     # Normalize logartithmic pitch between -1, 1
#     # vecs[1,...] = vecs[1,...].mean()
#     # if np.abs(vecs[1,...]).max() == 0:
#     #     vecs[1,...] = np.zeros(len(vecs[1,...]))
#     # else:
#     #     vecs[1,...] = vecs[1,...] / np.abs(vecs[1,...]).max()
#     vecs[...,1] = vecs[...,1].mean()
#     if np.abs(vecs[...,1]).max() == 0:
#         vecs[...,1] = np.zeros(len(vecs[...,1]))
#     else:
#         vecs[...,1] = vecs[...,1] / np.abs(vecs[...,1]).max()

#     return (offset, total,
#         np.hstack((
#             durations[..., None],
#             vecs)))


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

# def construct_teacher(notes, durations, indices):
#     min_note = notes[notes != 0].min()
#     #print(min_note)
#     note_vec_len = notes.max() - notes[notes != 0].min() + 1

#     min_dur = durations[1:].min()
#     max_dur = durations[1:].max()
#     dur_vec_len = max_dur - min_dur

#     Y = np.zeros((len(indices) - 1, note_vec_len + 1))

#     for (i, wi) in enumerate(indices[:-1]):
#         note_ohenc = np.zeros(note_vec_len)
#         if notes[wi+1] == 0:
#             note_ohenc[0] = 1
#         else:
#             note_ohenc[notes[wi+1]-min_note] = 1
#         dur = (durations[wi+1] - min_dur) / max_dur
#         Y[i,...] = np.hstack((note_ohenc, dur))

#     return (Y, (min_note, notes.max(), min_dur, max_dur))

# def do_something_with_y(Y, tprops):
#     min_note, max_note, min_dur, max_dur = tprops
#     print(Y.shape)
#     p = softmax(Y[:-1])
#     print('P:', p)
#     note = np.random.choice(
#         np.hstack((0, np.arange(min_note, max_note))),
#         p=p
#     )
#     sample_dur = Y[-1] * max_dur + min_dur
#     return (note, int(sample_dur))


if __name__ == "__main__":
    F = np.genfromtxt(r"C:\Users\Jesper\Documents\local study\1-2-ml\ml-"
                      + r"semester-project\data\F.txt", dtype=int)
    pass
