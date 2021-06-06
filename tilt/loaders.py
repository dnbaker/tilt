import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
try:
    from cytoolz import frequencies as Counter
except:
    from collections import Counter


def make_even_sampler(labels, idx=None, nsamples=None, replace=True):
    cts = Counter(labels[i] for i in (idx if idx else range(len(labels))))
    if not nsamples:
        nsamples = len(labels)
    return WeightedRandomSampler([1. / cts[i] for i in labels], nsamples, replacement=replace)


def make_biased_sampler(weights, nsamples=None, replace=True):
    if not nsamples:
        nsamples = len(weights)
    return WeightedRandomSampler(weights, nsamples, replacement=replace)


def make_weights(grouping_lists):
    counts = list(map(Counter, grouping_lists))
    freqs = [{k: 1. / v for k, v in c.item()} for c in counts]
    t = []
    for c, f, l in zip(counts, freqs, grouping_lists):
        t.append(np.array([f[l[i]] for i in range(len(grouping_lists[0]))]))
    return np.mean(np.vstack(t), axis=0)

## TODO:
##  Auto-contrastive pair/triple generation
##  Use coresets for selecting SGD methods
