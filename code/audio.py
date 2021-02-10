import argparse
import numpy as np
import math
from scipy.io.wavfile import write
from mido import Message, MidiTrack, MidiFile, MetaMessage,\
                 bpm2tempo, second2tick

BASE_FREQ = 440  # Hz
SAMPLE_RATE = 10000  # Samples per second
SYMBOL_DURATION = 1/20  # Seconds per symbol
TICKS_PER_SYMBOL = math.floor(SAMPLE_RATE * SYMBOL_DURATION)
BASE_KEY = 54
# Average non-zero symbol, keep music centered around
# base frequency


def process_voice(voice, **kwargs):
    bf = kwargs.get('base_frequency') or BASE_FREQ
    bk = kwargs.get('base_key') or BASE_KEY
    dur = kwargs.get('symbol_duration') or SYMBOL_DURATION
    rate = kwargs.get('sample_rate') or SAMPLE_RATE
    tps = math.floor(rate * dur)
    assert tps > 0

    out = np.zeros(voice.shape[0] * tps)
    curr_symbol = voice[0]
    start_idx = 0
    for i in range(voice.shape[0]):
        if voice[i] != curr_symbol:
            stop_idx = i-1
            tone_length = (stop_idx-start_idx) * tps
            freq = 0 if curr_symbol == 0 else \
                bf * 2**((curr_symbol-bk) / 12)
            out[start_idx*tps:stop_idx*tps] =\
                [math.sin(2 * math.pi * freq * (t+1) / rate) for
                 t in range(tone_length)]
            curr_symbol = voice[i]
            start_idx = i
    return out


def get_audio_vector(vec, voices=[0], **kwargs):
    samples_voice = \
        np.array([process_voice(vec[:, voice], **kwargs) for voice in
                  voices or range(vec.shape[1])])
    samples = np.sum(samples_voice, axis=0) / samples_voice.shape[0]
    m = np.max(np.abs(samples))
    return (samples/m).astype(np.float32)


def inferences_to_samples(inferences, dur_predict):
    out = []
    for inf in inferences:
        out = np.append(out, np.repeat(inf.note, inf.duration))
    return out[:dur_predict]


def save_inferences_to_midi(inferences, filename='Contrapunctus_XIV.mid'):
    print('Producing Midi file...')
    outfile = MidiFile()
    temp = bpm2tempo(48)  # or 76?
    # print('ticks_per_beat:', outfile.ticks_per_beat)
    outfile.ticks_per_beat = 2496

    for voice in range(len(inferences)):
        track = MidiTrack()
        outfile.tracks.append(track)
        track.append(MetaMessage('set_tempo', tempo=temp))
        track.append(Message('program_change', program=1))

        for inf in inferences[voice]:
            t = int(second2tick(inf.duration / 10.0, outfile.ticks_per_beat,
                                temp))
            track.append(Message('note_on', velocity=64, note=inf.note,
                                 time=t if inf.note == 0 else 0))
            track.append(Message('note_off', velocity=64, note=inf.note,
                                 time=0 if inf.note == 0 else t))

    outfile.save(filename)
    print('MidiFile saved...')
    return filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument(
        '-v', '--voices', type=int, nargs='+', required=False)
    parser.add_argument('out', type=str)
    args = parser.parse_args()

    vec = np.genfromtxt(args.input, delimiter=',')

    samples_f32 = get_audio_vector(vec, args.voices)

    sformat = args.out.split('.')[-1]
    if sformat == 'wav':
        write(args.out, SAMPLE_RATE, samples_f32)
    else:
        with open(args.out, 'w') as file:
            file.write(samples_f32)
