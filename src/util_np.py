import numpy as np

def sample(n, seed= 0):
    """yields samples from `n` nats."""
    data = list(range(n))
    while True:
        np.random.seed(seed)
        np.random.shuffle(data)
        yield from data
