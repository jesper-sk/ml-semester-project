# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn

# imports
import argparse
from datetime import datetime
import os

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
    parser.add_argument(
        '-v', '--voices', type=int, nargs='+', default=None,
        help='what voices to train on and generate for'
    )
    parser.add_argument(
        '-d', '--duration', type=int, required=False, default=400,
        help='amount of samples to predict'
    )
    parser.add_argument(
        '-f', '--file', type=str, required=False, default='audio',
        help='output file name'
    )
    parser.add_argument(
        '-p', '--path', type=str, required=False, default='../../experiments',
        help='the path where the experiment results will be saved'
    )
    parser.add_argument(
        '-a', '--alphas', type=float, nargs='+',
        help='the alpha values that will be evaluated during cross-validation'
    )
    parser.add_argument(
        '-A', '--alpharange', type=float, nargs=3,
        help='the range (from to increment) of alphas to evaluate'
    )
    parser.add_argument(
        '-w', '--windows', type=int, nargs='+',
        help='the window sizes that will be evaluated during cross-validation'
    )
    parser.add_argument(
        '-W', '--windowrange', type=int, nargs=3,
        help='the range (from to increment) of window sizes to evaluate'
    )
    parser.add_argument(
        '-s', '--sampler', type=str, nargs='+',
        help='the sampler(s) to use during inferencing, selecting multiple '
             'results in multiple output files per iteration'
    )
    parser.add_argument(
        '-P', '--plot', required=False, action="store_true",
        help='whether to show a plot'
    )
    parser.add_argument(
        '-O', '--offset', type=int, default=0, required=False
    )
    parser.add_argument(
        '-m', '--midi', required=False, action='store_true',
        help='Also output inferences as midi file'
    )
    parser.add_argument(
        '--noaudio', required=False, action='store_true'
    )
    args = parser.parse_args()

    voices = args.voices or [0]
    dur_predict = args.duration
    out_file = args.file

    sampler_entries = {
        'linear': TeacherGenerator.sample_linear,
        'softmax': TeacherGenerator.sample_softmax,
        'argmax': TeacherGenerator.take_argmax
    }
    samplers = args.sampler or ['linear']

    alpha_base = args.alphas or []
    alphas = np.arange(*args.alpharange).tolist() + alpha_base if \
        args.alpharange else args.alphas or [1.5]
    # Maybe single, most optimal alpha as default?

    window_base = args.windows or []
    windows = np.arange(*args.windowrange).tolist() + window_base if \
        args.windowrange else args.windows or [100]  # Idem dito

    print('\ntraining models for voices:', voices)
    print('with samplers:', samplers)
    print('and alphas:', alphas)
    print('and window sizes:', windows)

    raw_input = np.genfromtxt('../data/F.txt', dtype=int)
    raw_input = raw_input[args.offset:, ...]

    date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    dir = '%s/exp_%s' % (args.path, date)
    os.makedirs(dir, exist_ok=True)

    print('\nsaving to folder:', dir)

    out = {sampler: None for sampler in samplers}
    all_voice_inferences = {sampler: [] for sampler in samplers}

    log = np.array(
        ['voice', 'experiment', 'alpha', 'window size', 'mean score']
    ).reshape(1, -1)

    for voice in voices:
        print('\n-------- VOICE %s --------' % voice)

        # Transform data to input and teacher matrices
        notes, durations = transform.encode_duration(raw_input, voice)
        features = FeatureGenerator.construct_features(notes, durations)

        # X, indices = transform.windowed(features, window_size=windows[0])
        # Y = TeacherGenerator.construct_teacher(notes, durations, indices)

        # Train a ridge regression model
        # lr = obtain_optimal_model(X[:-1, ...], Y, alphas)
        X, _, lr, nlog = obtain_optimal_model(
            features, notes, durations, alphas, windows, log, voice)

        log = nlog

        for sampler in samplers:
            inferences = make_inferences(
                lr, X[-1, ...], dur_predict, sampler_entries[sampler]
            )
            samples = audio.inferences_to_samples(inferences, dur_predict)
            all_voice_inferences[sampler].append(inferences)

            # Add current voice inference to total
            if out[sampler] is None:
                out[sampler] = np.array(samples).reshape(1, -1).T
            else:
                out[sampler] = np.hstack(
                    (out[sampler], np.array(samples).reshape(1, -1).T)
                )

            with open('%s/inferences_%s_voice%s.csv' %
                      (dir, sampler, voice), 'w') as file:
                file.write(
                    '\n'.join(
                        ['%s,%s' % (inf.note, inf.duration)
                         for inf in inferences]
                    )
                )

            if not args.noaudio:
                write(
                    '%s/%s_%s_voice%s.wav' % (dir, out_file, sampler, voice),
                    rate=audio.SAMPLE_RATE,
                    data=audio.get_audio_vector(
                        np.array(samples).reshape(1, -1).T)
                )

            if args.plot:
                visualize_notes(samples, raw_input[-dur_predict:, voice])

    with open('%s/result_cross_validation.csv' % dir, 'w') as file:
        file.write('\n'.join([','.join(row) for row in log]))

    if len(voices) >= 2 and not args.noaudio:
        for sampler in samplers:
            audio_out = audio.get_audio_vector(out[sampler], voices=voices)
            write(
                "%s/%s_%s.wav" % (dir, out_file, sampler), data=audio_out,
                rate=audio.SAMPLE_RATE
            )

    if args.midi:
        for sampler in samplers:
            midi_file = audio.save_inferences_to_midi(
                all_voice_inferences[sampler], '%s/%s_%s.mid' %
                (dir, sampler, out_file)
            )

    # Enjoy some eargasming Bach!
    print('\n--------- DONE! ---------\n')
