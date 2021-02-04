import math
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
        print('min_dur:', cls._min_dur, 'max_dur:', cls._max_dur)
        # dur_vec_len = cls._max_dur - cls._min_dur        
        cls._init = True

        Y = np.zeros((len(indices) - 1, note_vec_len + 19))

        for (i, wi) in enumerate(indices[:-1]):
            note_ohenc = np.zeros(note_vec_len)
            if notes[wi+1] == 0:
                note_ohenc[0] = 1
            else:
                note_ohenc[notes[wi+1]-cls._min_note] = 1
            #dur = (durations[wi+1] - cls._min_dur) / cls._max_dur

            dur_ohenc = np.zeros(19)
            if durations[wi+1] > 36:
                dur_ohenc[18] = 1
            else:
                dur_ohenc[durations[wi+1]//2] = 1

            Y[i, ...] = np.hstack((note_ohenc, dur_ohenc))

        return Y

    @classmethod
    def y_to_note_dur(cls, Y):
        assert cls._init
        p1 = softmax(Y[:-19])
        note = np.random.choice(
            np.hstack((0, np.arange(cls._min_note, cls._max_note))),
            p=p1
        )

        # p2 = softmax(Y[-19:])
        # dur = np.random.choice(
        #     np.hstack((1, np.arange(1, 36, 2))),
        #     p=p2
        # )

        ind = np.argmax(Y[-19:])
        dur = np.hstack((1, np.arange(2, 37, 2)))[ind]
        # print(p2)

        # sample_dur = Y[-1] * cls._max_dur + cls._min_dur
        return (int(note), int(dur))  # int(math.ceil(sample_dur)))
