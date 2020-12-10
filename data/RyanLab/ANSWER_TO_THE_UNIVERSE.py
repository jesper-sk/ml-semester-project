
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style("white")
import sys
np.set_printoptions(threshold=sys.maxsize)



def get_data():

    bach = pd.read_csv("bach.csv", sep='\t')
    bach.columns = ['voice_1', 'voice_2', 'voice_3', 'voice_4']

    answer_to_the_universe = (bach["voice_4"][10])

    v1 = bach["voice_1"]
    v2 = bach["voice_2"]
    v3 = bach["voice_3"]
    v4 = bach["voice_4"]

    voices = v1, v2, v3, v4

    return voices


def multi_hist(voices, bin_count):

    v1, v2, v3, v4 = voices
    bin_count = bin_count
    
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].hist(v1, density = True, color = 'tab:blue', bins = bin_count)
    axs[0, 0].set_title('Voice 1')
    axs[0, 1].hist(v2, density = True, color = 'tab:green', bins = bin_count)
    axs[0, 1].set_title('Voice 2')
    axs[1, 0].hist(v3, density = True, color = 'tab:red', bins = bin_count)
    axs[1, 0].set_title('Voice 3')
    axs[1, 1].hist(v4, density = True, color = 'tab:orange', bins = bin_count)
    axs[1, 1].set_title('Voice 4')

    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')

    #Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.show()


def histo_pitch(voices):

    v1, v2, v3, v4 = voices
    kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})

    plt.figure(figsize=(10,7), dpi= 80)

    sns.distplot(v1, color="dodgerblue", label="Voice 1", **kwargs)
    sns.distplot(v2, color="orange", label="Voice 2", **kwargs)
    sns.distplot(v3, color="deeppink", label="Voice 3", **kwargs)
    sns.distplot(v4, color="mediumseagreen", label="Voice 4", **kwargs)
    #plt.xlim(0,12)
    #plt.ylim(0,.12)
    plt.title("Polyphonic Tabulation")
    plt.xlabel("Pitch")
    plt.ylabel("Density")
    plt.legend()

    plt.show()


def collapse(voices):

    vollapsed = np.zeros((len(voices), len(voices[0])))

    vm, vn = vollapsed.shape

    for n in range(vn):
        for m in range(vm):
            vollapsed[m][n] = voices[m][n]%11

    return vollapsed


def geen_null(voices):

    v_null_1 = []
    v_null_2 = []
    v_null_3 = []
    v_null_4 = []
    v_null_full = v_null_1, v_null_2, v_null_3, v_null_4

    for m in range(len(voices)):
        for n in range(len(voices[0])):
            if voices[m][n] != 0:
                v_null_full[m].append(voices[m][n]) 

    return v_null_full


def semi_tonal(voices):

    d_series = np.zeros((len(voices), len(voices[0])))
    vm, vn = d_series.shape

    for m in range(vm):
        for n in range(vn):
            if n != 0:
                d_series[m][n] = voices[m][n] - voices[m][n-1]

    return d_series

def pitch_sans_length(voices):

    l_series = np.zeros((len(voices), len(voices[0])))
    vm, vn = l_series.shape

    for m in range(vm):
        for n in range(vn):
            if n != 0:
                if voices[m][n] != voices[m][n-1]:
                    l_series[m][n] = voices[m][n]

    return l_series


def main():

    voices = get_data()

    vol = collapse(voices)

    vull_pitch = geen_null(vol)

    d_series = semi_tonal(voices)

    vull_tonal = geen_null(d_series)

    histo_pitch(vull_pitch)

    multi_hist(vull_pitch, 12)

    histo_pitch(vull_tonal)

    multi_hist(vull_tonal)

    l_series = pitch_sans_length(voices)

    vol_l =collapse(l_series)

    vol_lull = geen_null(vol_l)

    histo_pitch(vol_lull)

    multi_hist(vol_lull, 12)


main()

