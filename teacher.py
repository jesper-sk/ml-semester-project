import numpy as np
from scipy.special import softmax
from scipy.stats import norm


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
    def sample_linear(cls, range, p, prev_notes=None, min_note=None):
        p[p < 0] = 0
        p /= p.sum()
        # if prev_notes:
        #     if prev_notes[2] > 0:
                
                # # voice 0 range = (54,76)
                # # voice 1  = (45,71)
                # # voice 2 range = (40,62)
                # # voice 3 range = (28,54)
                
                # #v3
                # if min_note == 28:
                #     oct = 0

                # #v2
                # elif min_note == 40:
                #     oct = 12

                # #v1
                # elif min_note == 45:
                #     oct = 12

                # #v0
                # else:
                #     oct = 12*2

                # if prev_notes[2] == 38 + oct - min_note:
                #     pdf = norm.pdf(np.arange(1,len(p)+1,1), loc = 37 + oct - min_note, scale = 1)

                # elif prev_notes[2] == 37 + oct - min_note and prev_notes[1] == 38 + oct - min_note:
                #     pdf = norm.pdf(np.arange(1,len(p)+1,1), loc = 40 + oct - min_note, scale = 1)

                # elif prev_notes[2] == 40 + oct - min_note and prev_notes[1] == 37 + oct - min_note and prev_notes[0] == 38 + oct - min_note:
                #     pdf = norm.pdf(np.arange(1,len(p)+1,1), loc = 39 + oct - min_note, scale = 1)

                # else:
                #     pdf = norm.pdf(np.arange(1,len(p)+1,1), loc = prev_notes[2]+1 , scale = 1)

                # p = p*(pdf*100)
                # p[prev_notes[2]] = 0
                # p /= p.sum()


        return np.random.choice(range, p=p)

    @classmethod
    def sample_softmax(cls, range, p):
        p = softmax(p)
        print("softmax = ", p)
        return np.random.choice(range, p=p)

    @classmethod
    def take_argmax(cls, range, p):
        return range[np.argmax(p)]

    @classmethod
    def y_to_note_dur(cls, Y, prev_notes=None, min_note=None, sampler=sample_linear):
        assert cls._init
        if prev_notes:
            note = sampler(
                np.hstack((0, np.arange(cls._min_note, cls._max_note))),
                Y[:-19], prev_notes, min_note
            )
        else:
            note = sampler(
                np.hstack((0, np.arange(cls._min_note, cls._max_note))),
                Y[:-19]
            )
        #print(note)

        dur = sampler(
            np.hstack((1, np.arange(1, 36, 2))),
            Y[-19:]
        )
        dur = np.round(dur*.5)

        
        # sample_dur = Y[-1] * cls._max_dur + cls._min_dur
        return (int(note), int(dur))  # int(math.ceil(sample_dur)))
