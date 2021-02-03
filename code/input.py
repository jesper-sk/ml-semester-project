import math
import numpy as np

BASE_KEY = 54 # Key that denotes the base frequency
BASE_FREQ = 440 # Hz

CHROMA = [1,2,3,4,5,6,7,8,9,10,11,12]
CHROMA_R = 1

C5 = [1,8,3,10,5,12,7,2,9,4,11,6]
C5_R = 1

def note_to_vector(x, offset, total):
    if x == 0: return np.repeat(0, 5)
    else:
        min_note = offset
        max_note = offset + total - 1

        note = (x-55) % 12

        chroma_rad = (CHROMA[note] - 1) * (math.pi/6) # 2pi / 12
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

def encode_duration(F, voice=0):
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

def construct_features(notes, durations):
    offset = np.min(notes[notes!=0])
    total = len(np.unique(notes)) - 1 # remove 0

    vecs = np.array([note_to_vector(note, offset, total) for note in notes])

    # Normalize durations between -1, 1
    durations = durations - durations.mean()
    if np.abs(durations).max() == 0:
        durations = np.zeros(len(durations))
    else:
        durations = durations / np.abs(durations).max()
    
    # Normalize logartithmic pitch between -1, 1
    vecs[1,...] = vecs[1,...].mean()
    if np.abs(vecs[1,...]).max() == 0:
        vecs[1,...] = np.zeros(len(vecs[1,...]))
    else:
        vecs[1,...] = vecs[1,...] / np.abs(vecs[1,...]).max()

    return (offset, total, 
        np.hstack((
            durations[..., None],
            vecs)))

def biased(X):
    return np.hstack(
        (np.ones((len(X), 1)), X)
    )

def raw_to_features(F, voice=0):
    voice = F[..., voice]

    prev = voice[0]
    dur = 0

    offset = np.min(voice[voice!=0])
    total = len(np.unique(voice)) - 1

    durs = []
    vecs = [note_to_vector(voice[0], offset, total)]

    for note in voice:
        if note == prev:
            dur += 1
        else:
            vecs.append(note_to_vector(note, offset, total))
            durs.append(dur)
            prev = note
            dur = 1

    durs.append(dur)

    durs = np.array(durs)
    vecs = np.array(vecs)

    # Normalize each column between -1, 1
    durs = durs - durs.mean()
    if np.abs(durs).max() == 0:
        durs = np.zeros(durs)
    else:
        durs = durs / np.abs(durs).max()
    
    vecs[1,...] = vecs[1,...].mean()
    vecs[1,...] = vecs[1,...] / np.abs(vecs[1,...]).max()

    # Return
    out = np.hstack((np.ones(vecs.shape[1]), durs[...,None], vecs))
    return out

def windowed(X, window_size=10, hop_size=1):
    hop_size = hop_size or window_size
    offset = X.shape[0] % (window_size - hop_size)

    indices = range(offset+window_size-1, X.shape[0])\
        [0 : X.shape[0]-offset : hop_size]

    X_windows = np.array([X[(i-window_size):i, ...].flatten() for i in indices])

    return (X_windows, indices)

if __name__ == "__main__":
    F = np.genfromtxt(r"C:\Users\Jesper\Documents\local study\1-2-ml\ml-semester-project\data\F.txt", dtype=int)

    X, _ = windowed(raw_to_features(F))
    print(X.shape)