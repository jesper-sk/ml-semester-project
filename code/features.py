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

        res[..., 1] -= cls._logpitch_mean
        if cls._logpitch_amax == 0:
            res[..., 1] = np.zeros(len(res[..., 1]))
        else:
            res[..., 1] /= cls._logpitch_amax

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
             for note in notes]
        )

        cls._duration_mean = durations.mean()
        cls._duration_amax = np.abs(durations).max()
        cls._logpitch_mean = vecs[..., 1].mean()
        cls._logpitch_amax = np.abs(vecs[..., 1]).max()

        cls._init = True

        features = np.hstack((durations[..., None], vecs))
        return cls.feature_normalised(features)

        # Normalize durations between -1, 1
        # durations -= cls._duration_mean
        # if cls._duration_amax == 0:
        #     durations = np.zeros(len(durations))
        # else:
        #     durations /= cls._duration_amax

        # # Normalize logartithmic pitch between -1, 1
        # # vecs[1,...] = vecs[1,...].mean()
        # # if np.abs(vecs[1,...]).max() == 0:
        # #     vecs[1,...] = np.zeros(len(vecs[1,...]))
        # # else:
        # #     vecs[1,...] = vecs[1,...] / np.abs(vecs[1,...]).max()
        # vecs[...,1] -= cls._logpitch_amax
        # if cls._logpitch_amax == 0:
        #     vecs[...,1] = np.zeros(len(vecs[...,1]))
        # else:
        #     vecs[...,1] /= cls._logpitch_amax
    @classmethod
    def construct_chord_features(cls, chords, durations):
        cls._note_offset = np.min(chords[chords != 0])
        cls._note_total = len(np.unique(chords)) - 1

        vecs = np.array(
            [note_to_vector(note, cls._note_offset, cls._note_total)
             for note in chords]
        )

        cls._init = True

        features = np.hstack((durations[..., None], vecs))
        return features

    @classmethod
    def construct_single_feature(cls, note, duration, normalize=True):
        assert cls._init
        vector = note_to_vector(
            note,
            cls._note_offset,
            cls._note_total)
        feature = np.hstack((duration, vector))
        return cls.feature_normalised(feature) if normalize else feature

    @classmethod
    def feature_to_note_dur(cls, feat):
        dur = feat[0] * cls._duration_amax
        dur += cls._duration_mean

        vec = feat[1:]
        vec[0] *= cls._logpitch_amax
        vec[0] += cls._logpitch_mean

        return dur, transform.vec_to_note(vec, cls._note_offset, cls._note_total)
        