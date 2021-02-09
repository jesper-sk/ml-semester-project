# imports
import argparse
import sys

import numpy as np
from scipy.io.wavfile import write

import audio
from features import FeatureGenerator
from teacher import TeacherGenerator
import transform
from linear_regression import obtain_optimal_model, make_inferences
from visualize import visualize_notes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--voice', type=int, default=0)
    parser.add_argument('-V', '--voices', type=int, nargs='+', default=None)
    parser.add_argument('-d', '--duration', type=int, required=False,
                        default=400, help='How many samples to predict')
    parser.add_argument('-o', '--out', type=str, required=False,
                        default='Contrapunctus_XIV',
                        help='output desination')
    parser.add_argument('-a', '--alpha', type=float, nargs='+')
    parser.add_argument('-A', '--alpharange', type=float, nargs=3)
    parser.add_argument('-w', '--window', type=int, nargs='+')
    parser.add_argument('-W', '--windowrange', type=int, nargs=3)
    parser.add_argument('-s', '--sampler', type=str, nargs='+')
    parser.add_argument('-p', '--plot', required=False, action="store_true")
    parser.add_argument('-O', '--offset', type=int, default=0, required=False)
    parser.add_argument('-m', '--midi', required=False, action='store_true',
                        help='Also output inferences as midi file')
    parser.add_argument('--vmidi', required=False, action='store_true',
                        help='Visualize Midi')
    parser.add_argument('--noaudio', required=False, action='store_true')
    args = parser.parse_args()

    voices = args.voices or [args.voice]
    dur_predict = args.duration
    out_file = args.out

    sampler_entries = {
        'linear':TeacherGenerator.sample_linear,
        'softmax':TeacherGenerator.sample_softmax,
        'argmax':TeacherGenerator.take_argmax
    }
    samplers = [
        sampler_entries[sampler] for sampler in args.sampler or ['linear']
    ]

    alphas = np.arange(*args.alpharange) if args.alpharange else \
            args.alpha or [0, .1, .25, .5, .8, 1, 1.5] # Maybe single, most optimal alpha as default?

    windows = np.arang(*args.windowrange) if args.windowrange else \
            args.window or [42] # Idem dito

    print('\ntraining models for voices:', voices)
    print('with samplers:', args.sampler)
    print('and alphas:', alphas)
    print('and window sizes:', windows)

    raw_input = np.genfromtxt('../data/F.txt', dtype=int)

    raw_input = raw_input[args.offset:, ...]

    out = None
    all_voice_inferences = []
    for voice in voices:
        print('\n-------- VOICE %s --------' % voice)
        
        # Transform data to input and teacher matrices
        notes, durations = transform.encode_duration(raw_input, voice)
        features = FeatureGenerator.construct_features(notes, durations)
        X, indices = transform.windowed(features, window_size=windows[0])
        Y = TeacherGenerator.construct_teacher(notes, durations, indices)
        print("features shape:", X.shape)
        print("teacher shape:", Y.shape)

        # Train a ridge regression model    
        lr = obtain_optimal_model(X[:-1, ...], Y, alphas)
        inferences = make_inferences(lr, X[-1, ...], dur_predict, samplers[0])
        samples = audio.inferences_to_samples(inferences, dur_predict)
        all_voice_inferences.append(inferences)

        # Add current voice inference to total
        if out is None:
            out = np.array(samples).reshape(1,-1).T
            print(out.shape)
        else:
            out = np.hstack((out, np.array(samples).reshape(1,-1).T))

        if not args.noaudio:
            write(
                '%s (voice %s).wav' % (out_file, voice), rate=audio.SAMPLE_RATE,
                data=audio.get_audio_vector(np.array(samples).reshape(1,-1).T)
            )

        if args.plot:
            visualize_notes(samples, raw_input[-dur_predict:, voice])

    if len(voices) >= 2 and not args.noaudio:
        audio_out = audio.get_audio_vector(out, voices=voices)
        write("%s.wav" % out_file, data=audio_out, rate=audio.SAMPLE_RATE)

    if args.midi or args.vmidi:
        midi_file = audio.save_inferences_to_midi(all_voice_inferences,
                                                 '%s.mid' % out_file)

    # Enjoy some eargasming Bach!
    print('\n-------- DONE --------\n')
