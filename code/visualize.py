import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def visualize_notes(notes, original=None):
    plt.rcParams.update({'font.size': 20})
    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14) 

    if original is not None:
        _, ax = plt.subplots()
        plt.subplots_adjust(left=.05, right=.95)
        ax.scatter(
            np.arange(original.shape[0]), original,
            c='red', s=10)
        ax.scatter(
            np.arange(original.shape[0], original.shape[0]+notes.shape[0]),
            notes, c='green', s=10)
        ax.set_ylim(-2, 100)
        ax.set_xlim(-10, original.shape[0]+notes.shape[0]+10)
        ax.grid(True, alpha=.25)
        ax.set_title('Last and new 400 samples Bach')
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Note')
    else:
        _, ax = plt.subplots()
        ax.scatter(np.arange(notes.shape[0]), notes)
    plt.show()

