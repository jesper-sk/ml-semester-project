

import pandas as pd

bach = pd.read_csv("bach.csv", sep='\t')
bach.columns = ['voice_1', 'voice_2', 'voice_3', 'voice_4']

answer_to_the_universe = (bach["voice_4"][10])

print (answer_to_the_universe)

