import numpy as np

from note_to_vector import note_to_vector

raw_input = np.genfromtxt('../data/F.txt')
unique = np.unique(raw_input)
# print(len(unique), unique)

notes = [np.array(["raw", "pitch", "chroma_x", "chroma_y", "c5_x", "c5_y"])]
for u in unique[1:]:
    vec = note_to_vector(int(u), 28, 62)
    vec = np.insert(vec, 0, u, axis=0)
    notes.append(vec)
    print(u, '=', vec)

np.savetxt('test.csv', np.asarray(notes), fmt='%s', delimiter=',')
# np.min(f[f!=0][0, ...])
# np.unique(F[F!=0][0, ...])
