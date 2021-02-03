import numpy as np
from scipy.special import softmax

class TeacherGenerator:
    _init = False
    _min_note = 0
    _max_note = 0
    _min_dur = 0
    _max_dur = 0

    @classmethod
    def construct_teacher(cls, notes, durations, indices):
        cls._min_note = notes[notes != 0].min()
        cls._max_note = notes.max()
        note_vec_len = cls._max_note - cls._min_note + 1

        cls._min_dur = durations[1:].min()
        cls._max_dur = durations[1:].max()
        dur_vec_len = cls._max_dur - cls._min_dur

        cls._init = True

        Y = np.zeros((len(indices) - 1, note_vec_len + 1))

        for (i, wi) in enumerate(indices[:-1]):
            note_ohenc = np.zeros(note_vec_len)
            if notes[wi+1] == 0:
                note_ohenc[0] = 1
            else:
                note_ohenc[notes[wi+1]-cls._min_note] = 1
            dur = (durations[wi+1] - cls._min_dur) / cls._max_dur
            Y[i,...] = np.hstack((note_ohenc, dur))

        return Y

    @classmethod
    def do_something_with_y(cls, Y):
        assert cls._init
        p = softmax(Y[:-1])
        note = np.random.choice(
            np.hstack((0, np.arange(cls._min_note, cls._max_note))),
            p=p
        )
        sample_dur = Y[-1] * cls._max_dur + cls._min_dur
        return (int(note), int(sample_dur))
