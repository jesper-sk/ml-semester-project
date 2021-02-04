import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import make_scorer

from features import FeatureGenerator
from teacher import TeacherGenerator


class Inference:
    def __init__(self, values):
        self.note, self.duration = values


def obtain_optimal_model(X, Y, alphas):
    best_alpha = None
    best_mean_score = -np.inf
    for a in alphas:
        scores = []
        lr = Ridge(alpha=a)
        scores = cross_val_score(
            lr, X, Y, cv=X.shape[0],
            scoring=make_scorer(custom_scorer, greater_is_better=True)
        )
        print('alpha:', a, 'mean:', scores.mean())
        # print('scores:', scores)
        if scores.mean() > best_mean_score:
            best_alpha = a
            best_mean_score = scores.mean()
    assert best_alpha is not None
    print("Best alpha:", best_alpha, 'Best mean score:', best_mean_score)
    best_lr = Ridge(best_alpha)
    best_lr.fit(X, Y)
    return best_lr


def custom_scorer(Y_true, Y_pred, **kwargs):
    note_true, dur_true = TeacherGenerator.y_to_note_dur(
        Y_true.squeeze())
    note_pred, dur_pred = TeacherGenerator.y_to_note_dur(
        Y_pred.squeeze())

    feat_true = FeatureGenerator.construct_single_feature(note_true, dur_true)
    feat_pred = FeatureGenerator.construct_single_feature(note_pred, dur_pred)
    # x = [1, 2]
    # y = [3, 4]
    #> d = sqrt(1^2 - 3^2) + sqrt(2^2 - 4^2) = sqrt(9-1) + sqrt(16-4) = sqrt(8) + sqrt(12) = 7

    # d = sqrt(sum((x-y)^2))
    # d = sqrt((1-3)^2 + (2-4)^2) = sqrt(4+4) = sqrt(8)
    # MSE = sum((x-y)^2) / N
    # SSE = sum((x-y)^2)
    # x = -b +- sqrt(b^2 - 4ac)/2a

    # sqrt(sum((x-y)^2))
    return -np.sqrt(np.sum(np.square(feat_true-feat_pred)))


def make_inferences(lr, x, dur_predict):
    inferences = []
    last_x = x
    while get_inferenced_time(inferences) < dur_predict:
        Y = lr.predict(x.reshape(1, -1))
        inference = Inference(TeacherGenerator.y_to_note_dur(Y.squeeze()))

        if inference.duration < 1 and inferences:
            Y = lr.predict(last_x.reshape(1, -1))
            inf = Inference(TeacherGenerator.y_to_note_dur(
                Y.squeeze()
            ))
            x[-6:, ...] = FeatureGenerator.construct_single_feature(
                inf.note, inf.duration
            )
            inferences = inferences[:-1]
        else:
            inference.duration = max(inference.duration, 1)
            inferences.append(inference)
            last_x = x
            x = np.hstack((
                x[6:, ...],
                FeatureGenerator.construct_single_feature(
                    inference.note, inference.duration
                )
            ))
            assert last_x is not x
        inf_time = get_inferenced_time(inferences)
        progress = ['=' for i in range(int(inf_time/dur_predict*100-1))]
        print('{0:0=3d}'.format(inf_time),
              ''.join(progress) + '>'
              + ''.join([' ' for i in range(100-len(progress))]) + '|')
    
    out = np.array([])
    for inf in inferences:
        out = np.append(out, np.repeat(inf.note, inf.duration))
    return out


def get_inferenced_time(inferences):
    return np.sum(i.duration for i in inferences)
