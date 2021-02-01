import numpy as np
import math

BASE_KEY = 54 # Key that denotes the base frequency
BASE_FREQ = 440 # Hz

def note_to_vector(x, offset, total):
    if x == 0: return np.repeat(0, 5)
    else:
        min_note = offset
        max_note = offset + total - 1

        chroma = [1,2,3,4,5,6,7,8,9,10,11,12]
        chroma_r = 1

        c5 = [1,8,3,10,5,12,7,2,9,4,11,6]
        c5_r = 1

        note = (x-55 % 12)

        chroma_rad = (chroma[note] - 1) * (math.pi/6) # 2pi / 12
        c5_rad = (c5[note] - 1) * (math.pi/6)

        chroma_x = chroma_r * math.sin(chroma_rad)
        chroma_y = chroma_r * math.cos(chroma_rad)

        c5_x = c5_r * math.sin(c5_rad)
        c5_y = c5_r * math.cos(c5_rad)

        n = x - BASE_KEY
        freq = 2**(n/12) * BASE_FREQ

        min_p = 2 * math.log2(2**((min_note - BASE_KEY)/12) * BASE_FREQ)
        max_p = 2 * math.log2(2**((max_note - BASE_KEY)/12) * BASE_FREQ)

        pitch = 2 * math.log2(freq) - max_p + ((max_p - min_p)/2)

        return np.array(pitch, chroma_x, chroma_y, c5_x, c5_y)