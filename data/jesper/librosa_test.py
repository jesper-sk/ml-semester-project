# I don't recommend to run this thing, a lot of plots then. Plots can be commented out if necessary
import numpy as np
import librosa
import librosa.display
import librosa.feature
import matplotlib
import matplotlib.pyplot as plt

from scipy.spatial import distance
from transform import get_audio_vector

MODE = 'save' # 'save' or 'load'
VOICES = None # None or list of voices

def specplot(y, xl, yl):
    fig, ax = plt.subplots(figsize=(10,8))
    D = librosa.amplitude_to_db(np.abs(y), ref=np.max)

def compute_ssm(X, metric="seuclidean"):
    """Computes the self-similarity matrix of X."""
    D = distance.pdist(X.T, metric=metric)
    D = distance.squareform(D)
    D /= D.max()
    return 1 - D

def compute_sm(Xa, Xb, metric='seuclidean'):
    D = distance.cdist(Xa, Xb, metric)
    D /= D.max()
    return 1 - D

sr=10000

if MODE=='save':
    vec = np.genfromtxt('../F.txt')
    audio = get_audio_vector(vec, VOICES)
    np.save('F_samples.npy', audio)
else:
    audio = np.load('F_samples.npy')

fig, ax = plt.subplots()
librosa.display.waveplot(audio, sr=10000, ax=ax)
plt.show()

stft = librosa.feature.chroma_stft(audio, sr=sr)
D = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
fig, ax = plt.subplots(figsize=(10,8))
librosa.display.specshow(D, ax=ax, sr=sr, y_axis='linear', x_axis='time')
plt.show()

cqt = librosa.amplitude_to_db(np.abs(librosa.feature.chroma_cqt(audio, sr=sr))**2, ref=np.median)
fig, ax = plt.subplots()
librosa.display.specshow(cqt, ax=ax, x_axis='time', y_axis='linear')
plt.show()

f, ax = plt.subplots(figsize=(10,10))
ax.matshow(compute_ssm(cqt, 'correlation'), interpolation='nearest', cmap='plasma')

samples = [get_audio_vector(vec, [voice]) for voice in [0,1,2,3]]
for voice in samples:
    cqt = librosa.amplitude_to_db(np.abs(
        librosa.feature.chroma_cqt(voice, sr=sr))**2, ref=np.median)
    ssm = compute_ssm(cqt, 'seuclidean')
    f, ax = plt.subplots(figsize=(10,10))
    ax.matshow(ssm, interpolation='nearest', cmap='plasma')

for voice1 in samples:
    for voice2 in samples:
        cqt1 = librosa.amplitude_to_db(np.abs(
            librosa.feature.chroma_cqt(voice1, sr=sr))**2, ref=np.median)
        cqt2 = librosa.amplitude_to_db(np.abs(
            librosa.feature.chroma_cqt(voice2, sr=sr))**2, ref=np.median)
        ssm = compute_sm(cqt1.T, cqt2.T, 'seuclidean')
        f, ax = plt.subplots(figsize=(10,10))
        ax.matshow(ssm, interpolation='nearest', cmap='plasma')