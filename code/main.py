# imports
import argparse
import numpy as np
from scipy.io.wavfile import write

from inference import make_inferences
from input import raw_to_features, windowed
from linear_regression import obtain_optimal_weights
from output import construct_output
from transform import get_audio_vector


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('voice', type=int)
    args = parser.parse_args()

    voice = args.voice

    raw_input = np.genfromtxt('../data/F.txt')


    raw_input = np.ndarray.astype(raw_input, int)
    


    # Convert raw_input to actual feature vectors
    # Figure something out so that we can use windowed input
        # (over multiple differently sized windows)
    features = raw_to_features(raw_input)

    X, indices = windowed(features)

    #Y = construct_output(raw_input, indices, voice)


    # TODO (See linear_regression.py) Obtain the weight vector/matrix
    weights = obtain_optimal_weights(X,features)

    # TODO (See inference.py) Predict the next 20+ seconds
    #inferences = make_inferences(weights, X[-1, ...], 20)

    # Concatenate inferences to original data
    # (assume voice is 0-indexed by the user)
    #raw_out = np.append(np.array(raw_input[voice]), np.array(inferences))

    # Convert our wonderfully smart output into a wave file
    # TODO: Prepare raw_out to be compatible with get_audio_vector
    #audio_out = get_audio_vector(raw_out)
    #write('Contrapunctus_XIV.wav', data=raw_out, rate=10000)

    # Enjoy some eargasming Bach!
