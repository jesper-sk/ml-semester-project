import math
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
    # best_a = None
    # best_w = None
    # best_mean_score = -np.inf
    models = np.array([-np.inf, None, None, None, None]).reshape(1,-1)
    for (i, (a, w)) in enumerate(product(alphas, windows)):
        print('alpha=%s\twindow=%s' % (a, w), end='')
        X, indices = transform.windowed(features, window_size=w)
        Y = TeacherGenerator.construct_teacher(notes, durations, indices)

        lr = Ridge(alpha=a, normalize=normalize)
        scores = cross_val_score(
            lr, X[:-1, ...], Y, cv=X.shape[0]-1,
            scoring=make_scorer(custom_scorer, greater_is_better=True)
        )
        ms = scores.mean()
        print('\tmean_score=%s' % ms)
        log = np.append(
            log,
            np.array(
                [str(voice), str(i), str(a), str(w), str(ms)]
            ).reshape(1, -1),
            axis=0
        )
        if math.isnan(ms):
            continue

        lr.fit(X[:-1, ...], Y)
        models = np.append(
            models,
            np.array(
                [ms, a, w, lr, X]
            ).reshape(1,-1),
            axis=0
        )

        # if scores.mean() > best_mean_score:
        #     best_a, best_w = a, w
        #     best_mean_score = scores.mean()
    #print(models)
    models = models[1:,...]
    sorted = models[models[..., 0].argsort()]   
    top5 = sorted[-5:] if len(sorted) >= 5 else sorted
    print('----------top %s----------' % len(top5))
    for i,top in enumerate(top5):
        assert top[0] != -np.inf
        print(len(top5)-i, "a=%s, w=%s, score=%s" %
            (top[1], top[2], top[0]))

    # out_lr = Ridge(best_a, normalize=normalize)
    # X, indices = transform.windowed(features, window_size=best_w)
    # Y = TeacherGenerator.construct_teacher(notes, durations, indices)
    # out_lr.fit(X[:-1, ...], Y)
    return top5, log


def custom_scorer(Y_true, Y_pred, **kwargs):
    note_true, dur_true = TeacherGenerator.y_to_note_dur(
        Y_true.squeeze(), sampler=TeacherGenerator.take_argmax)
    note_pred, dur_pred = TeacherGenerator.y_to_note_dur(
        Y_pred.squeeze(), sampler=TeacherGenerator.take_argmax)

    feat_true = FeatureGenerator.construct_single_feature(note_true, dur_true)
    feat_pred = FeatureGenerator.construct_single_feature(note_pred, dur_pred)
    return -np.sqrt(np.sum(np.square(feat_true-feat_pred)))

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
