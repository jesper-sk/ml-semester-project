from mido import Message, MidiFile, MidiTrack

def inferences_to_midi(inferences):
    outfile = MidiFile()
    for _ in inferences.shape[1]:
        track = MidiTrack()
        outfile.tracks.append(track)
        track.append(Message('program_change', program=12))

        for inf in inferences:
            track.append(Message('note_on', note=inf.note, time=inf.duration))
            track.append(Message('note_off', note=inf.note, time=inf.duration))

    outfile.save('test.mid')
