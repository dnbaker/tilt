Simple biased samplers for PyTorch


For class-balanced samplers:
```python
from tilt import make_even_sampler, make_biased_sampler

sampler = make_even_sampler(labels) # Makes a class-balanced sampler
sampler = make_even_sampler(labels, idx=[1,4, 5]) # Makes a class-balanced sampler over the points 1, 4, 5
sampler = make_even_sampler(labels, nsamples=10000) # Makes a sampler with 10,000 samples
```
For class-balanced samplers:
```python
from tilt import make_even_sampler, make_biased_sampler

weights = np.abs(np.random.standard_cauchy((1000,)))
sampler = make_biased_sampler(weights, nsamples=100000)
```

Then, pass this as the sampler for PyTorch DataLoader construction.

In the future, I plan to add in contrastive learning mining utilities and autoaugmentation. For both of these, LSH techniques might come into play.
