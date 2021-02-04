# imports
import argparse

import sys
import numpy as np
from scipy.io.wavfile import write

from features import FeatureGenerator
from teacher import TeacherGenerator
import transform
from linear_regression import obtain_optimal_model, make_inferences
from audio import get_audio_vector
from visualize import visualize_notes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--voice', type=int, default=0)
    parser.add_argument('-d', '--duration', type=int, required=False,
                        default=400, help='How many samples to predict')
    parser.add_argument('-o', '--out', type=str, required=False,
                        default='Contrapunctus_XIV.wav',
                        help='output desination')
    args = parser.parse_args()

    voice = args.voice
    dur_predict = args.duration
    out_file = args.out

    raw_input = np.genfromtxt('../data/F.txt', dtype=int)
    # raw_input = np.ndarray.astype(raw_input, int)

    # Convert raw_input to actual feature vectors
    # Figure something out so that we can use windowed input
    # (over multiple differently sized windows)
    notes, durations = transform.encode_duration(raw_input, voice)
    features = FeatureGenerator.construct_features(notes, durations)
    X, indices = transform.windowed(features)
    # X = transform.biased(X)
    Y = TeacherGenerator.construct_teacher(notes, durations, indices)

    # Train a ridge regression model
    alphas = [0, .1, .25, .5, .75, 1, 1.25, 1.5]
    # alphas = [1]
    lr = obtain_optimal_model(X[:-1, ...], Y, alphas)
    inferences = make_inferences(lr, X[-1, ...], dur_predict)

    # Concatenate inferences to original data
    # (assume voice is 0-indexed by the user)
    # raw_out = np.array(inferences).reshape(1, -1).T
    # raw_out = np.append(np.array(raw_input[..., voice]),
    #                     np.array(inferences))
    print(raw_input[-400:, voice].shape)
    print(np.array(inferences).shape)
    raw_out = np.hstack((raw_input[-400:, voice], np.array(inferences)))\
        .reshape(1, -1)
    print(raw_out.shape)

    # Convert our wonderfully smart output into a wave file
    audio_out = get_audio_vector(np.array(inferences).reshape(1, -1).T,
                                 [voice])
    write(out_file, data=audio_out, rate=10000)

    write('combined.wav', data=get_audio_vector(raw_out.T, [voice]),
          rate=10000)

    visualize_notes(inferences, raw_input[-400:, voice])

    # Enjoy some eargasming Bach!

    # MAYBE NOT NECESSARY
    # TODO (See linear_regression.py) Obtain the weight vector/matrix
    # weights = obtain_optimal_weights(X,features)
    # TODO (See inference.py) Predict the next 20+ seconds
    # inferences = make_inferences(weights, X[-1, ...], 20)
