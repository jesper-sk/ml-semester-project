import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics.scorer import make_scorer

#from transform import construct_features, do_something_with_y
from features import FeatureGenerator
from teacher import TeacherGenerator

# def obtain_optimal_weights(X,Y):
#     W = np.zeros((60,1))
#     steps = len(Y)-len(X)
#     for i in range(steps):
#         x = X[i,:].reshape(60,1)
#         y = Y[i+10+1,:].reshape(6,1)
#         w = np.dot(np.linalg.pinv(np.dot(x,np.transpose(x))),x)
#         for j in range(10):
#             for jj in range(6):
#                 w[j+jj] = w[j+jj]*y[jj]
#         W += w
#     W = W/steps
#     print(W) 
#     return W

def obtain_optimal_model(X, Y, alphas):
    best_alpha = None
    best_mean_score = -np.inf
    loo = LeaveOneOut()
    for a in alphas:
        scores = []
        # lr = Ridge(alpha=a)
        # for train_index, test_index in loo.split(X):          
        #     lr.fit(X[train_index,...], Y[train_index,...])
        #     score = lr.score(X[test_index,...], Y[test_index,...])
        #     print('alpha:', a, 'score:', score)
        #     scores.append(score)

        lr = Ridge(alpha=a)
        scores = cross_val_score(
            lr, X, Y, cv=X.shape[0], 
            scoring=make_scorer(custom_scorer, greater_is_better=False)
        )
        print('alpha:', a)#, 'scores:', scores)
        if scores.mean() > best_mean_score:
            best_alpha = a
            best_mean_score = scores.mean()
    assert best_alpha != None
    best_lr = Ridge(best_alpha)
    best_lr.fit(X,Y)
    return best_lr

def custom_scorer(Y_true, Y_pred, **kwargs):
    note_true, dur_true = TeacherGenerator.do_something_with_y(Y_true.squeeze())
    note_pred, dur_pred = TeacherGenerator.do_something_with_y(Y_pred.squeeze())

    feat_true = FeatureGenerator.construct_single_feature(note_true, dur_true)
    feat_pred = FeatureGenerator.construct_single_feature(note_pred, dur_pred)
    return -np.sum(np.square(feat_true, feat_pred))

def make_inferences(lr, x, dur_predict):
    inferences = np.array([])
    inferenced_time = 0 # In seconds
    while inferenced_time < dur_predict:
        Y = lr.predict(x.reshape(1,-1))
        print('Y:', Y)
        note, sample_dur = TeacherGenerator.do_something_with_y(Y.squeeze())
        sample_dur = max(sample_dur, 1)
        print('Note:', note, 'Sample_dur:', sample_dur)
        np.append(inferences, np.repeat(note, sample_dur))
        inferenced_time += sample_dur # TODO make sure dur_sec represents seconds
        x = np.hstack((
            x[6:,...], 
            FeatureGenerator.construct_single_feature(note, sample_dur)
        ))
    return inferences

# def make_inferences(W, x, sec_duration):
#     inferences = []
#     inferenced_time = 0 # In seconds
#     while inferenced_time < sec_duration:
#         Y = W @ x
#         # TODO make Y one_hot_encoded
#         inferences.append(Y)
#         inferenced_time += dur_to_seconds(Y[-1])
#         x = np.hstack((x[6:,...], to_feature(Y)))
    
#     return inferences
