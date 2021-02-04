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

# from mido import Message, MidiFile, MidiTrack
# def convert_to_midi(notes):
#     outfile = MidiFile(type=0)
#     track = MidiTrack()
#     outfile.tracks.append(track)

#     track.append(Message('program_change', program=12, time=0))
#     track.append(Message('note_on', ))
