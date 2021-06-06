import torch
import numpy as np
try:
    from cytoolz import frequencies as Counter
except:
    from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler

def make_even_sampler(labels, idx=None, nsamples=None, replace=True):
    cts = Counter(labels[i] for i in (idx if idx else range(len(labels))))
    if not nsamples:
        nsamples = len(labels)
    freqs = {k: 1. / v for k, v in cts.items()}
    w = np.array([freqs[i] for i in labels])
    return WeightedRandomSampler(np.array([freqs[i] for i in labels]),
                                 nsamples, replacement=replace)


def make_biased_sampler(weights, nsamples=None, replace=True):
    if not nsamples:
        nsamples = len(weights)
    return WeightedRandomSampler(weights, nsamples, replacement=replace)

## TODO:
##  Auto-contrastive pair/triple generation
##  Use coresets for selecting SGD methods
