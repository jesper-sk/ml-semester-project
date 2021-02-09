import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import make_scorer

from features import FeatureGenerator
from teacher import TeacherGenerator


class Inference:
    def __init__(self, values):
        self.note, self.duration = values


def obtain_optimal_model(X, Y, alphas, normalize=False):
    best_alpha = None
    best_mean_score = -np.inf
    for a in alphas:
        scores = []
        lr = Ridge(alpha=a, normalize=normalize)
        scores = cross_val_score(
            lr, X, Y, cv=X.shape[0],
            scoring=make_scorer(custom_scorer, greater_is_better=True)
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
    
    return -np.sqrt(np.sum(np.square(feat_true-feat_pred)))


def make_inferences(lr, x, dur_predict, sampler):
    inferences = []
    while get_inferenced_time(inferences) < dur_predict:
        Y = lr.predict(x.reshape(1, -1))
        inference = Inference(
            TeacherGenerator.y_to_note_dur(
                Y.squeeze(), sampler=sampler
            )
        )
        inferences.append(inference)
        x = np.hstack((
            x[6:, ...],
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
