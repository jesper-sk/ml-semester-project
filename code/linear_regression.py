import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import make_scorer

from features import FeatureGenerator
from teacher import TeacherGenerator


def obtain_optimal_model(X, Y, alphas):
    best_alpha = None
    best_mean_score = -np.inf
    for a in alphas:
        scores = []
        lr = Ridge(alpha=a)
        scores = cross_val_score(
            lr, X, Y, cv=X.shape[0],
            scoring=make_scorer(custom_scorer, greater_is_better=False)
        )
        print('alpha:', a)
        if scores.mean() > best_mean_score:
            best_alpha = a
            best_mean_score = scores.mean()
    assert best_alpha is not None
    best_lr = Ridge(best_alpha)
    best_lr.fit(X, Y)
    return best_lr


def custom_scorer(Y_true, Y_pred, **kwargs):
    note_true, dur_true = TeacherGenerator.do_something_with_y(
        Y_true.squeeze())
    note_pred, dur_pred = TeacherGenerator.do_something_with_y(
        Y_pred.squeeze())

    feat_true = FeatureGenerator.construct_single_feature(note_true, dur_true)
    feat_pred = FeatureGenerator.construct_single_feature(note_pred, dur_pred)
    return -np.sum(np.square(feat_true, feat_pred))


def make_inferences(lr, x, dur_predict):
    inferences = np.array([])
    inferenced_time = 0  # In seconds
    while inferenced_time < dur_predict:
        Y = lr.predict(x.reshape(1, -1))
        # print('Y:', Y)
        note, sample_dur = TeacherGenerator.do_something_with_y(Y.squeeze())
        sample_dur = max(sample_dur, 1)
        print('Note:', note, 'Sample_dur:', sample_dur)
        inferences = np.append(inferences, np.repeat(note, sample_dur))
        inferenced_time += sample_dur
        x = np.hstack((
            x[6:, ...],
            FeatureGenerator.construct_single_feature(note, sample_dur)
        ))
    return inferences
