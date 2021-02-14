import numpy as np
from itertools import product
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

from features import FeatureGenerator
from teacher import TeacherGenerator
import transform

from scipy.stats import norm


class Inference:
    def __init__(self, values):
        self.note, self.duration = values


def obtain_optimal_model(features, notes, durations, alphas, windows, log,
                         voice, normalize=False):
    best_a = None
    best_w = None
    best_mean_score = -np.inf
    for (i, (a, w)) in enumerate(product(alphas, windows)):
        print('alpha=%s\twindow=%s' % (a, w), end='')
        X, indices = transform.windowed(features, window_size=w)
        Y = TeacherGenerator.construct_teacher(notes, durations, indices)

        lr = Ridge(alpha=a, normalize=normalize)
        scores = cross_val_score(
            lr, X[:-1, ...], Y, cv=X.shape[0]-1,
            scoring=make_scorer(custom_scorer, greater_is_better=False)
        )
        print('\tmean_score=%s' % scores.mean())
        log = np.append(
            log,
            np.array(
                [str(voice), str(i), str(a), str(w), str(scores.mean())]
            ).reshape(1, -1),
            axis=0
        )

        if scores.mean() > best_mean_score:
            best_a, best_w = a, w
            best_mean_score = scores.mean()

    assert best_a is not None
    print('-------------------------')
    print('best pair: a=%s, w=%s, score=%s' %
          (best_a, best_w, best_mean_score))

    out_lr = Ridge(best_a, normalize=normalize)
    X, indices = transform.windowed(features, window_size=best_w)
    Y = TeacherGenerator.construct_teacher(notes, durations, indices)
    out_lr.fit(X[:-1, ...], Y)
    return X, indices, out_lr, log


def obtain_optimal_model_old(X, Y, alphas, normalize=False):
    best_alpha = None
    best_mean_score = -np.inf
    for a in alphas:
        scores = []
        lr = Ridge(alpha=a, normalize=normalize)
        scores = cross_val_score(
            lr, X, Y, cv=X.shape[0],
            scoring=make_scorer(custom_scorer, greater_is_better=False)
        )
        print('alpha:', a, 'mean:', scores.mean())
        if scores.mean() > best_mean_score:
            best_alpha = a
            best_mean_score = scores.mean()
    assert best_alpha is not None
    print("Best alpha:", best_alpha, 'Best mean score:', best_mean_score)
    best_lr = Ridge(best_alpha, normalize=normalize)
    best_lr.fit(X, Y)
    return best_lr


def custom_scorer(Y_true, Y_pred, **kwargs):
    note_true, dur_true = TeacherGenerator.y_to_note_dur(
        Y_true.squeeze(), sampler=TeacherGenerator.take_argmax)
    note_pred, dur_pred = TeacherGenerator.y_to_note_dur(
        Y_pred.squeeze(), sampler=TeacherGenerator.take_argmax)

    feat_true = FeatureGenerator.construct_single_feature(note_true, dur_true)
    feat_pred = FeatureGenerator.construct_single_feature(note_pred, dur_pred)
    return np.sqrt(np.sum(np.square(feat_true-feat_pred)))


def make_inferences(lr, X, dur_predict, sampler):
    inferences = []

    while get_inferenced_time(inferences) < dur_predict:
        Y = lr.predict(X.reshape(1, -1))


        if len(inferences) < 4:
            inference = Inference(
                TeacherGenerator.y_to_note_dur(
                    Y.squeeze(), sampler=sampler)
                )

        else:
            prev_notes = []
            min_note = TeacherGenerator._min_note
            for i in range(3):
                prev_notes.append(inferences[-4+i].note - min_note + 1)

            P = Y.squeeze()
            p = P[:-19]
            d = np.ones(len(P)-len(p))

            # voice 0 range = (54,76)
            # voice 1  = (45,71)
            # voice 2 range = (40,62)
            # voice 3 range = (28,54)
            
            if prev_notes[2] > 0:
                #v3
                if min_note == 28:
                    oct = 0
                #v2
                elif min_note == 40:
                    oct = 12
                #v1
                elif min_note == 45:
                    oct = 12
                #v0
                else:
                    oct = 12*2

                if prev_notes[2] == 38 + oct - min_note:
                    pdf = norm.pdf(np.arange(1,len(p)+1,1), loc = 37 + oct - min_note, scale = 1)

                elif prev_notes[2] == 37 + oct - min_note and prev_notes[1] == 38 + oct - min_note:
                    pdf = norm.pdf(np.arange(1,len(p)+1,1), loc = 40 + oct - min_note, scale = 1)

                elif prev_notes[2] == 40 + oct - min_note and prev_notes[1] == 37 + oct - min_note and prev_notes[0] == 38 + oct - min_note:
                    pdf = norm.pdf(np.arange(1,len(p)+1,1), loc = 39 + oct - min_note, scale = 1)

                else:
                    pdf = norm.pdf(np.arange(1,len(p)+1,1), loc = prev_notes[2]+1 , scale = 1)

                p = p*(pdf*100)
                p[prev_notes[2]] = 0

            P = np.concatenate((p,d))

            inference = Inference(
                TeacherGenerator.y_to_note_dur(
                    P, prev_notes, TeacherGenerator._min_note, sampler=sampler)
                )
            

        inferences.append(inference)
        X = np.hstack((
            X[6:, ...],
            FeatureGenerator.construct_single_feature(
                    inference.note, inference.duration
            )
        ))
    # out = np.array([])
    # for inf in inferences:
    #     out = np.append(out, np.repeat(inf.note, inf.duration))
    return inferences


def get_inferenced_time(inferences):
    return np.sum(i.duration for i in inferences)
