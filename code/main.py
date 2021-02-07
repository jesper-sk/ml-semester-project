# imports
import argparse
import sys

import numpy as np
from scipy.io.wavfile import write

from features import FeatureGenerator
from teacher import TeacherGenerator
import transform
from linear_regression import obtain_optimal_model, make_inferences
from audio import get_audio_vector, save_inferences_to_midi,\
                  inferences_to_samples
from visualize import visualize_notes
from roll import visualize_midi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--voice', type=int, default=0)
    parser.add_argument('--voices', type=int, nargs='+', default=None)
    parser.add_argument('-d', '--duration', type=int, required=False,
                        default=400, help='How many samples to predict')
    parser.add_argument('--audio', type=bool, required=False, default=True)
    parser.add_argument('-o', '--out', type=str, required=False,
                        default='Contrapunctus_XIV',
                        help='output desination')
    parser.add_argument('-p', '--plot', required=False, action="store_true")
    parser.add_argument('-w', '--window_size', type=int, 
                        default=42) # It is the answer to the universe after all
    parser.add_argument('--offset', type=int, default=0, required=False)
    parser.add_argument('--sampler', type=str, default='linear')
    parser.add_argument('--single_model', action='store_true', required=False)
    parser.add_argument('--midi', required=False, action='store_true',
                        help='Also output inferences as midi file')
    parser.add_argument('--vmidi', required=False, action='store_true',
                        help='Visualize Midi')
    args = parser.parse_args()

    voices = args.voices or [args.voice]
    dur_predict = args.duration
    out_file = args.out

    sampler = {
        'linear':TeacherGenerator.sample_linear,
        'softmax':TeacherGenerator.sample_softmax,
        'argmax':TeacherGenerator.take_argmax
    }[args.sampler]

    raw_input = np.genfromtxt('../data/F.txt', dtype=int)
    # raw_input = np.ndarray.astype(raw_input, int)

    raw_input = raw_input[args.offset:, ...]

    if args.single_model:
        
        chords, durations = transform.encode_duration(raw_input)
        features = FeatureGenerator.construct_chord_features(chords, durations)
        X, indices = transform.windowed(
            features, window_size=args.window_size
        )
        print("Signal shape: %s" % str(X.shape))

        # TODO: PCA if n>=N?

        Y = TeacherGenerator.construct_chord_teacher(chords, durations, indices)
        print("Teacher shape:", Y.shape)

        sys.exit()

    out = None
    all_voice_inferences = []
    for voice in voices:

        notes, durations = transform.encode_duration(raw_input, voice)
        features = FeatureGenerator.construct_features(notes, durations)
        X, indices = transform.windowed(
            features, window_size=args.window_size
        )
        print("features shape: %s" % str(X.shape))
        Y = TeacherGenerator.construct_teacher(notes, durations, indices)

        # Train a ridge regression model
        alphas = [.1, .25, .5, .75, 1, 1.25, 1.5]
        lr = obtain_optimal_model(X[:-1, ...], Y, alphas)
        inferences = make_inferences(
            lr, X[-1, ...], dur_predict, sampler)
        samples = inferences_to_samples(inferences, dur_predict)
        all_voice_inferences.append(inferences)

        # Add current voice inference to total
        if out is None:
            out = np.array(samples).reshape(1,-1).T
            print(out.shape)
        else:
            out = np.hstack((out, np.array(samples).reshape(1,-1).T))

        if args.audio:
            write(
                '%s (voice %s).wav' % (out_file, voice), rate=10000,
                data=get_audio_vector(np.array(samples).reshape(1,-1).T)
            )

        if args.plot:
            visualize_notes(samples, raw_input[-dur_predict:, voice])

    if len(voices) >= 2 and args.audio:
        audio_out = get_audio_vector(out, voices=voices)
        write("%s.wav" % out_file, data=audio_out, rate=10000)

    if args.midi or args.vmidi:
        midi_file = save_inferences_to_midi(all_voice_inferences,
                                            '%s.mid' % out_file)
        if args.vmidi:
            visualize_midi(midi_file)


    # Concatenate inferences to original data
    # (assume voice is 0-indexed by the user)
    # raw_out = np.array(inferences).reshape(1, -1).T
    # raw_out = np.append(np.array(raw_input[..., voice]),
    #                     np.array(inferences))
    # print(raw_input[-400:, voice].shape)
    # print(np.array(inferences).shape)
    # raw_out = np.hstack((raw_input[-400:, voice], np.array(inferences)))\
    #     .reshape(1, -1)
    # print(raw_out.shape)

    # Convert predicted notes into a wave file
    # audio_out = get_audio_vector(np.array(inferences).reshape(1, -1).T)
    # write(out_file, data=audio_out, rate=10000)

    # write('combined.wav', data=get_audio_vector(raw_out.T, [voice]),
    #       rate=10000)
    # 

    # Enjoy some eargasming Bach!
    print("=== DONE ===")
