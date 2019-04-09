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
if False:
    backward_prob = backward(example['obsloglik'],
            np.log(wordHMMs['o']["startprob"]),
            np.log(wordHMMs['o']["transmat"]))

    print(np.sum(np.abs(backward_prob)))
    print(np.sum(np.abs(example["logbeta"])))

##### Section 6.1
if False:
    forward_prob = forward(example['obsloglik'],
            np.log(wordHMMs['o']["startprob"]),
            np.log(wordHMMs['o']["transmat"]))
    backward_prob = backward(example['obsloglik'],
            np.log(wordHMMs['o']["startprob"]),
            np.log(wordHMMs['o']["transmat"]))
    state_posterior = statePosteriors(forward_prob, backward_prob)

    GMM_state_posterior = np.zeros(state_posterior.shape)
    HMM = wordHMMs['o']
    print(HMM["means"].shape)
    GMM_state_posterior = log_multivariate_normal_density_diag(
           example['lmfcc'], HMM["means"], HMM["covars"])
    for i in range(GMM_state_posterior.shape[0]):
        GMM_state_posterior[i, :] = GMM_state_posterior[i, :] - logsumexp(GMM_state_posterior[i, :])

    print(np.sum(np.exp(GMM_state_posterior), axis = 1))

##### Section 6.2

phoneHMMs = all_model['phoneHMMs'].item()

wordHMMs_all_model = {}
for key in isolated.keys():
    wordHMMs_all_model[key] = concatHMMs(phoneHMMs, isolated[key])

data_10 = data[10]
HMM = wordHMMs_all_model["9"]

max_iters = 20

means = HMM["means"]
covars = HMM["covars"]

log_likelihood = +np.inf
for i in range(max_iters):
    obs_log_lik = log_multivariate_normal_density_diag(data_10["lmfcc"],
                means,
                covars)

    forward_prob = forward(obs_log_lik,
                np.log(HMM["startprob"]),
                np.log(HMM["transmat"]))

    backward_prob = backward(obs_log_lik,
            np.log(HMM["startprob"]),
            np.log(HMM["transmat"]))

    log_likelihood_new = logsumexp(forward_prob[-1, :])

    if abs(log_likelihood_new - log_likelihood) < 1:
        break

    log_likelihood = log_likelihood_new
    print(log_likelihood)

    log_gamma = statePosteriors(forward_prob, backward_prob)

    means, covars = updateMeanAndVar(data_10["lmfcc"], log_gamma, varianceFloor=5.0)
