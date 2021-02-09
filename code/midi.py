import numpy as np
import mido
from audio import save_inferences_to_midi
from linear_regression import Inferences
from transform import encode_duration


if __name__ == "__main__":
    raw_input = np.genfromtxt('../data/F.txt', dtype=int)
    infs = []
    for v in range(4):
        voice = []
        for note, duration in encode_duration(raw_input, v):
            voice.append(Inference((note, duration)))
        infs.append(voice)
        