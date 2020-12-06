import argparse
import numpy as np
import sys, os
import math
from scipy.io.wavfile import write

BASE_FREQ = 440
SAMPLE_RATE = 10000
SYMBOL_DURATION = 1/20
TICKS_PER_SYMBOL = math.floor(SAMPLE_RATE * SYMBOL_DURATION)

def process_voice(voice, min_key_nr):
    out = np.zeros(voice.shape[0] * TICKS_PER_SYMBOL)
    curr_symbol = voice[0]
    start_idx = 0
    for i in range(voice.shape[0]):
        if voice[i] != curr_symbol:
            stop_idx = i-1
            tone_length = (stop_idx-start_idx) * TICKS_PER_SYMBOL
            freq = BASE_FREQ * 2**((curr_symbol-min_key_nr) / 12)
            out[start_idx*TICKS_PER_SYMBOL:stop_idx*TICKS_PER_SYMBOL] = \
                [math.sin(2 * math.pi * freq * (t+1) / SAMPLE_RATE) for \
                    t in range(tone_length)]
            curr_symbol = voice[i]
            start_idx = i
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument(
        '-v', '--voices', type=int, nargs='+', required=False)
    parser.add_argument('out', type=str)
    args = parser.parse_args()

    vec = np.genfromtxt(args.input, delimiter=',')
    min_key_nr = np.min(vec[np.nonzero(vec)])
    samples_voice = \
        np.array([process_voice(vec[:,voice], min_key_nr) for voice in \
                  args.voices or range(vec.shape[1])])
    samples = np.sum(samples_voice, axis=0) / samples_voice.shape[0]
    m = np.max(np.abs(samples))
    samples_f32 = (samples/m).astype(np.float32)
    
    sformat = args.out.split('.')[-1]
    if sformat == 'wav':
        write(args.out, SAMPLE_RATE, samples_f32)
    else:
        with open(args.out, 'w') as file:
            file.write(samples_f32)
