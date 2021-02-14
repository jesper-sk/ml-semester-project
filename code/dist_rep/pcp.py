
import numpy as np
import librosa
import random as rnd

# TODO: Work with masked numpy arrays? Such that silence = mask

def to_categorical(y, num_classes=None, dtype='float32'):
  """This method was copied from the Keras utils code to avoid loading in full
  Keras for one stupid little method. 
  
  Converts a class vector (integers) to binary class matrix.
  E.g. for use with categorical_crossentropy.
  Arguments:
      y: class vector to be converted into a matrix
          (integers from 0 to num_classes).
      num_classes: total number of classes. If `None`, this would be inferred
        as the (largest number in `y`) + 1.
      dtype: The data type expected by the input. Default: `'float32'`.
  Returns:
      A binary matrix representation of the input. The classes axis is placed
      last.
  Example:
  >>> a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
  >>> a = tf.constant(a, shape=[4, 4])
  >>> print(a)
  tf.Tensor(
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]], shape=(4, 4), dtype=float32)
  >>> b = tf.constant([.9, .04, .03, .03,
  ...                  .3, .45, .15, .13,
  ...                  .04, .01, .94, .05,
  ...                  .12, .21, .5, .17],
  ...                 shape=[4, 4])
  >>> loss = tf.keras.backend.categorical_crossentropy(a, b)
  >>> print(np.around(loss, 5))
  [0.10536 0.82807 0.1011  1.77196]
  >>> loss = tf.keras.backend.categorical_crossentropy(a, a)
  >>> print(np.around(loss, 5))
  [0. 0. 0. 0.]
  Raises:
      Value Error: If input contains string value
  """
  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  if not num_classes:
    num_classes = np.max(y) + 1
  n = y.shape[0]
  categorical = np.zeros((n, num_classes), dtype=dtype)
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (num_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical

def symbol_to_hz(x, base_freq=440, base_key_estimator=np.average):
    base_key = int(base_key_estimator(x[np.nonzero(x)]))
    f = np.array(x, copy=True)
    f[np.nonzero(f)] = base_freq * 2**(((f[np.nonzero(f)]-base_key))/12)
    return f.astype(int)

def hz_to_note_idx(x, octave_invariant=True):
    n = np.array([['' if item==0 else \
        librosa.hz_to_note(item, octave=not octave_invariant) \
            for item in row] for row in x]) 

    n_i = np.ndarray(n.shape, dtype=int)
    if octave_invariant:
        occ = ['', 'C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 
            'G', 'G♯', 'A', 'A♯', 'B'] 
    else:
        occ = [''] + np.array([ \
            [f'C{i}', f'C♯{i}', f'D{i}', f'D♯{i}', f'E{i}', f'F{i}', 
             f'F♯{i}', f'G{i}', f'G♯{i}', f'A{i}', f'A♯{i}', f'B{i}'] \
                for i in range(2,7)]).flatten().tolist()
    
    for (i, note) in enumerate(occ):
        n_i[np.where(n==note)] = i
                    
    return n_i

def hz_to_pcp(x, octave_invariant=True):
    n = np.array([['' if item==0 else \
        librosa.hz_to_note(item, octave=not octave_invariant) \
            for item in row] for row in x]) 

    n_i = np.ndarray(n.shape, dtype=int)
    if octave_invariant:
        occ = ['', 'C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 
            'G', 'G♯', 'A', 'A♯', 'B'] 
    else:
        occ = [''] + np.array([ \
            [f'C{i}', f'C♯{i}', f'D{i}', f'D♯{i}', f'E{i}', f'F{i}', 
             f'F♯{i}', f'G{i}', f'G♯{i}', f'A{i}', f'A♯{i}', f'B{i}'] \
                for i in range(2,7)]).flatten().tolist()
    
    for (i, note) in enumerate(occ):
        n_i[np.where(n==note)] = i

    n_i = np.vstack((np.repeat(np.array(range(len(occ)))[...,None], 
                                x.shape[1], axis=1), n_i))

    encoded = to_categorical(n_i[:,0])
    for v in range(1,x.shape[1]):
        encoded += to_categorical(n_i[:,v])

    encoded = encoded[len(occ):,...]

    encoded /= np.max(encoded)
    return occ, encoded

def symbol_to_pcp(x, octave_invariant=True, **kwargs):
    return hz_to_pcp(symbol_to_hz(x, **kwargs), octave_invariant)

def random_ohenc(l, n):
    rnd.seed()
    res = np.zeros(l)
    idcs = rnd.choices(list(range(l)), k=n)
    for idx in idcs:
        res[idx] += 1
    res /= n
    return res

def with_noise(pcp, n, n_v):
    return np.vstack((pcp, np.array([random_ohenc(pcp.shape[1], n_v) \
        for _ in range(n)]))) if n>0 else pcp

if __name__=='__main__':
    x = np.genfromtxt(r'..\..\data\F.txt')
    f = symbol_to_hz(x)
    label_o, pcp_o = hz_to_pcp(f, False)
    label, pcp = hz_to_pcp(f, True)
    np.save('npys/F-pcp.npy', pcp)
    np.save('npys/F-pcp-o.npy', pcp_o)
    np.save('npys/F-pcp-lab.npy', label)
    np.save('npys/F-pcp-o-lab.npy', label_o)
    np.save('npys/F-hz.npy', f)