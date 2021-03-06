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

        cls._min_dur = durations[notes != 0].min()
        cls._max_dur = durations[notes != 0].max()

        cls._init = True

        Y = np.zeros((len(indices) - 1, note_vec_len + 19))

        for (i, wi) in enumerate(indices[:-1]):
            note_ohenc = np.zeros(note_vec_len)
            if notes[wi+1] == 0:
                note_ohenc[0] = 1
            else:
                note_ohenc[notes[wi+1]-cls._min_note] = 1

            dur_ohenc = np.zeros(19)
            if durations[wi+1] > 36:
                dur_ohenc[18] = 1
            else:
                dur_ohenc[durations[wi+1]//2] = 1

            Y[i, ...] = np.hstack((note_ohenc, dur_ohenc))

        return Y

    @classmethod
    def sample_linear(cls, range, p):
        p[p < 0] = 0
        p /= p.sum()
        return np.random.choice(range, p=p)

    @classmethod
    def sample_softmax(cls, range, p):
        p = softmax(p)
        return np.random.choice(range, p=p)

    @classmethod
    def take_argmax(cls, range, p):
        return range[np.argmax(p)]

    @classmethod
    def y_to_note_dur(cls, Y, sampler=sample_linear):
        assert cls._init
        note = sampler(
            np.hstack((0, np.arange(cls._min_note, cls._max_note))),
            Y[:-19]
        )
        dur = sampler(
            np.hstack((1, np.arange(1, 36, 2))),
            Y[-19:]
        )
        # sample_dur = Y[-1] * cls._max_dur + cls._min_dur
        return (int(note), int(dur))  # int(math.ceil(sample_dur)))
