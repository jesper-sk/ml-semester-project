import numpy as np
from itertools import product
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.exceptions import FitFailedWarning

from features import FeatureGenerator
from teacher import TeacherGenerator
import transform

import warnings
warnings.filterwarnings("ignore", category=FitFailedWarning)


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


def custom_scorer(Y_true, Y_pred, **kwargs):
    note_true, dur_true = TeacherGenerator.y_to_note_dur(
        Y_true.squeeze(), sampler=TeacherGenerator.take_argmax)
    note_pred, dur_pred = TeacherGenerator.y_to_note_dur(
        Y_pred.squeeze(), sampler=TeacherGenerator.take_argmax)

    if note_true == 0 and note_pred!=0: 
        return 6
    if note_pred==0 and note_true!=0:
        return 6

    feat_true = FeatureGenerator.construct_single_feature(note_true, dur_true)
    feat_pred = FeatureGenerator.construct_single_feature(note_pred, dur_pred)
    return np.sqrt(np.sum(np.square(feat_true-feat_pred)))

def inform_output(Y, inferences):
    if len(inferences) == 0:
        return
    last = inferences[-1]
    last_note_idx = 0 if last.note == 0 else \
        last.note - TeacherGenerator._min_note
    Y[last_note_idx] = 0
    assert np.any(Y!=0)

def make_inferences(lr, X, dur_predict, sampler):
    inferences = []
    while get_inferenced_time(inferences) < dur_predict:
        Y = lr.predict(X.reshape(1, -1)).squeeze()
        inform_output(Y, inferences)
        inference = Inference(
            TeacherGenerator.y_to_note_dur(
                Y.squeeze(), sampler=sampler
            )
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
