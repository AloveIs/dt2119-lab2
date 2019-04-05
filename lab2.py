import numpy as np
from lab2_proto import *
from lab2_tools import *
from prondict import prondict
import matplotlib.pyplot as plt

data = np.load('lab2_data.npz')['data']

one_speaker_model = np.load('lab2_models_onespkr.npz')
all_model = np.load('lab2_models_all.npz')
example = np.load('lab2_example.npz')["example"]
example.shape=(1,)
example = example[0]
# load one model

isolated = {}
for digit in prondict.keys():
    isolated[digit] = ['sil'] + prondict[digit] + ['sil']


phoneHMMs = np.load('lab2_models_onespkr.npz')['phoneHMMs'].item()

isolated['o']=['sil', 'ow', 'sil']
isolated['3']=['sil', 'th', 'r', 'iy', 'sil']
wordHMMs = {}

wordHMMs['o'] = concatHMMs(phoneHMMs, isolated['o'])
wordHMMs['3'] = concatHMMs(phoneHMMs, isolated['3'])

# log_multivariate_normal_density_diag(X, means, covars):
# test correctness

if False:
    result = log_multivariate_normal_density_diag(example["lmfcc"],
                    wordHMMs['o']["means"],
                    wordHMMs['o']["covars"])
    print(np.sum(example["obsloglik"] - result))


result = log_multivariate_normal_density_diag(data[9]["lmfcc"],
            wordHMMs['3']["means"],
            wordHMMs['3']["covars"])

plt.pcolormesh(result.T)
plt.colorbar()
plt.show()



def print_data_idx(data):
    for i in range(len(data)):
        print(i," ",data[i]["digit"])
