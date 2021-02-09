import numpy as np
import mido
from audio import save_inferences_to_midi
from linear_regression import Inferences


if __name__ == "__main__":
    raw_input = np.genfromtxt('../data/F.txt', dtype=int)
    