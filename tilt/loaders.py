import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
try:
    from cytoolz import frequencies as Counter
except:
    from collections import Counter


def make_even_sampler(labels, idx=None, nsamples=None, replace=True):
    """
    make_even_sampler
        Args:
            labels: Sequence of labels
        Keyword Args:
            idx = None
                To only sample from specific indices, provide a list or numpy array of indices
            nsamples = None
                To change the number of samples, change this to an integer
            replace = True
                To avoid sampling with replacement, change this to False
                Guarantees for gradient sampling estimation are usually based on sampling *with* replacement.

        Returns:
            torch.utils.data.sampler.WeightedRandomSampler
    """
    cts = Counter(labels[i] for i in (idx if idx else range(len(labels))))
    if not nsamples:
        nsamples = len(labels)
    return WeightedRandomSampler([1. / cts[i] for i in labels], nsamples, replacement=replace)


def make_biased_sampler(weights, nsamples=None, replace=True):
    """
    make_biased_sampler
        Args:
            labels: Sequence of labels
        Keyword Args:
            nsamples = None
                To change the number of samples, change this to an integer
            replace = True
                To avoid sampling with replacement, change this to False
                Guarantees for gradient sampling estimation are usually based on sampling *with* replacement.

        Returns:
            torch.utils.data.sampler.WeightedRandomSampler
    """
    if not nsamples:
        nsamples = len(weights)
    return WeightedRandomSampler(weights, nsamples, replacement=replace)


def make_weights(grouping_lists):
    """
    make_weights
    Converts a list of grouping assignments into a set of sampling weights for a collection of data points

        Input: Iterable[Iterable[Int]]

        Output: numpy.ndarray, numpy.float64
    """
    counts = list(map(Counter, grouping_lists))
    t = []
    for c, l in zip(counts, grouping_lists):
        t.append(1. / np.array([c[l[i]] for i in range(len(grouping_lists[0]))]))
    ret = np.mean(np.vstack(t), axis=0)
    return ret / np.sum(ret)

## TODO:
##  Auto-contrastive pair/triple generation
##  Use coresets for selecting SGD methods

__all__ = ["make_even_sampler", "make_biased_sampler", "make_weights"]
