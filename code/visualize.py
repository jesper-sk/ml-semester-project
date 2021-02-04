import matplotlib.pyplot as plt
import numpy as np


def visualize_notes(notes, original=None):
    if original is not None:
        _, (ax1, ax2) = plt.subplots(2, 1)
        ax1.scatter(np.arange(notes.shape[0]), notes, 
                    c='green')
        ax1.set_title('New Bach')
        ax2.scatter(np.arange(original.shape[0]), original, 
                    c='orange')
        ax2.set_title('Original Bach')

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
