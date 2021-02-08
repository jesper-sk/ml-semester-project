import matplotlib.pyplot as plt
import numpy as np


def visualize_notes(notes, original=None):
    if original is not None:
        _, (og_ax, new_ax) = plt.subplots(2, 1)
        new_ax.scatter(np.arange(notes.shape[0]), notes,
                       c='green')
        new_ax.set_title('New Bach')
        new_ax.set_ylim(-2, 100)

        og_ax.scatter(np.arange(original.shape[0]), original,
                      c='orange')
        og_ax.set_title('Original Bach')
        og_ax.set_ylim(-2, 100)

    else:
        _, ax = plt.subplots()
        ax.scatter(np.arange(notes.shape[0]), notes)
    plt.show()

