import numpy as np
from transform import note_to_vector


class FeatureGenerator:
    _init = False
    _note_offset = 0
    _note_total = 0
    _logpitch_mean = 0
    _logpitch_amax = 0
    _duration_mean = 0
    _duration_amax = 0

    @classmethod
    def feature_normalised(cls, feature):
        assert cls._init

        res = feature.copy()

        # res[..., 1] -= cls._logpitch_mean
        # if cls._logpitch_amax == 0:
        #     res[..., 1] = np.zeros(len(res[..., 1]))
        # else:
        #     res[..., 1] /= cls._logpitch_amax

        res[..., 0] -= cls._duration_mean
        if cls._duration_amax == 0:
            res[..., 0] = np.zeros(len(res[..., 1]))
        else:
            res[..., 0] /= cls._duration_amax

        return res

    @classmethod
    def construct_features(cls, notes, durations):
        cls._note_offset = np.min(notes[notes != 0])
        cls._note_total = len(np.unique(notes)) - 1

        vecs = np.array(
            [note_to_vector(note, cls._note_offset, cls._note_total)
             for note in notes])

        cls._duration_mean = durations.mean()
        cls._duration_amax = np.abs(durations).max()
        cls._logpitch_mean = vecs[..., 1].mean()
        cls._logpitch_amax = np.abs(vecs[..., 1]).max()

        cls._init = True

        features = np.hstack((durations[..., None], vecs))
        return cls.feature_normalised(features)

    @classmethod
    def construct_single_feature(cls, note, duration, normalize=True):
        assert cls._init
        vector = note_to_vector(
            note,
            cls._note_offset,
            cls._note_total)
        feature = np.hstack((duration, vector))
        return cls.feature_normalised(feature) if normalize else feature
