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

wordHMMs = {}
for key in isolated.keys():
    wordHMMs[key] = concatHMMs(phoneHMMs, isolated[key])


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

#plt.pcolormesh(result.T)
#plt.colorbar()
#plt.show()

#def print_data_idx(data):
    #for i in range(len(data)):
        #print(i," ",data[i]["digit"])

##### Section 5.2
if False:
    forward_prob = forward(example['obsloglik'],
            np.log(wordHMMs['o']["startprob"]),
            np.log(wordHMMs['o']["transmat"]))

# FORWARD works
#print(forward_prob)
#print(example["logalpha"])

#print(logsumexp(forward_prob[-1, :]))
#print(example['loglik'])

if False:
    scores = np.zeros((44, 11))
    for i in range(len(data)):
        data_ = data[i]["lmfcc"]

        j = 0
        for key, HMM in wordHMMs.items():
            data_log_lik = log_multivariate_normal_density_diag(
                    data_, HMM["means"], HMM["covars"])
            fw = forward(data_log_lik,
                    np.log(HMM["startprob"]),
                    np.log(HMM["transmat"]))
            scores[i, j] = logsumexp(fw[-1, :])
            j += 1

    print(scores)

##### Section 5.3
if False:
    viterbi_loglik, viterbi_path = viterbi(example['obsloglik'],
            np.log(wordHMMs['o']["startprob"]),
            np.log(wordHMMs['o']["transmat"]))

    #print(viterbi_loglik)
    print(viterbi_loglik)
    print(example['vloglik'])

##### Section 5.4

backward_prob = backward(example['obsloglik'],
        np.log(wordHMMs['o']["startprob"]),
        np.log(wordHMMs['o']["transmat"]))

print(np.sum(np.abs(backward_prob)))
print(np.sum(np.abs(example["logbeta"])))
