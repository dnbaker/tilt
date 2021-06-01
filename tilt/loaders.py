import torch
import numpy as np
try:
    from cytoolz import frequencies as Counter
except:
    from collections import Counter
import torch.utils.data.sampler as torchsampler

def make_even_sampler(dataset, labels, idx=None, nsamples=None):
    cts = Counter(labels[i] for i in (idx if idx else range(len(labels))))
    if nsamples is None:
        nsamples = len(labels)
    freqs = {k: 1. / v for k, v in cts.items()}
    w = np.array([freqs[i] for i in labels])
    return sampler.WeightedRandomSampler(w, nsamples, replacement=true)



def make_biased_sampler(dataset, weights, nsamples=None):
    if nsamples is None:
        nsamples = len(labels)
    return sampler.WeightedRandomSampler(weights, nsamples, replacement=true)

## TODO:
##  Auto-contrastive pair/triple generation
##  Use coresets for selecting SGD methods
