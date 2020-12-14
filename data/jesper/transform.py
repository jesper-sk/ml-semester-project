import argparse
import numpy as np
import math
from scipy.io.wavfile import write

BASE_FREQ = 440 # Hz
SAMPLE_RATE = 10000 # Samples per second
SYMBOL_DURATION = 1/20 # Seconds per symbol
TICKS_PER_SYMBOL = math.floor(SAMPLE_RATE * SYMBOL_DURATION)
BASE_KEY = 54 # int(np.average(vec[np.nonzero(vec)]))

def process_voice(voice):
    out = np.zeros(voice.shape[0] * TICKS_PER_SYMBOL)
    curr_symbol = voice[0]
    start_idx = 0
    for i in range(voice.shape[0]):
        if voice[i] != curr_symbol:
            stop_idx = i-1
            tone_length = (stop_idx-start_idx) * TICKS_PER_SYMBOL
            freq = 0 if curr_symbol == 0 else \
                BASE_FREQ * 2**((curr_symbol-BASE_KEY) / 12)
            out[start_idx*TICKS_PER_SYMBOL:stop_idx*TICKS_PER_SYMBOL] = \
                [math.sin(2 * math.pi * freq * (t+1) / SAMPLE_RATE) for \
                    t in range(tone_length)]
            curr_symbol = voice[i]
            start_idx = i
    return out

def get_audio_vector(vec, voices=None):
    samples_voice = \
        np.array([process_voice(vec[:,voice]) for voice in \
                  voices or range(vec.shape[1])])
    samples = np.sum(samples_voice, axis=0) / samples_voice.shape[0]
    m = np.max(np.abs(samples))
    return (samples/m).astype(np.float32)

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
