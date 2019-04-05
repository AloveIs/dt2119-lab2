import numpy as np
from lab2_proto import *

data = np.load('lab2_data.npz')['data']

one_speaker_model = np.load('lab2_models_onespkr.npz')
all_model = np.load('lab2_models_all.npz')
example = np.load('lab2_example.npz')

# load one model

phoneHMMs = np.load('lab2_models_onespkr.npz')['phoneHMMs'].item()

print("Modeled phonemes:")
print(sorted(phoneHMMs.keys()))

print(phoneHMMs["ey"]["startprob"])
concatTwoHMMs(phoneHMMs["ey"], phoneHMMs["ah"])
