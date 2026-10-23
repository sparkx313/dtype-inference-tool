"""Bounded, reproducible sampling.

A column of 40 million rows does not need 40 million values to decide whether it
is a date. Reservoir sampling (Algorithm R) gives a uniform sample of `k` without
knowing the population in advance, so the cost of inference is set by `k` and not
by the file.
"""

import random


def reservoir_sample(values, k, seed=0):
    """A uniform sample of at most `k` items from an iterable of unknown length.

    The result keeps arrival order rather than sorted order, which makes it
    readable in a failure message. Same seed plus same input order gives the same
    sample; a different *population* size changes it, because the replacement
    decisions depend on how many items have been seen.
    """
    if k <= 0:
        raise ValueError("sample size must be positive")
    reservoir = []
    rng = random.Random(seed)
    for index, value in enumerate(values):
        if index < k:
            reservoir.append(value)
            continue
        replace = rng.randrange(index + 1)
        if replace < k:
            reservoir[replace] = value
    return reservoir


class ColumnSampler:
    """Simultaneous reservoirs for every column of a row stream.

    One pass over the file fills all of them, so the memory ceiling is
    `columns * k` values no matter how long the file is.
    """

    def __init__(self, k=1000, seed=0):
        if k <= 0:
            raise ValueError("sample size must be positive")
        self.k = k
        self.seed = seed
        self.rng = random.Random(seed)
        self.reservoirs = {}
        self.counts = {}
        self.rows = 0

    def add(self, row):
        self.rows += 1
        for column, value in row.items():
            seen = self.counts.get(column, 0) + 1
            self.counts[column] = seen
            reservoir = self.reservoirs.get(column)
            if reservoir is None:
                reservoir = self.reservoirs[column] = []
            if len(reservoir) < self.k:
                reservoir.append(value)
            elif self.rng.randrange(seen) < self.k:
                reservoir[self.rng.randrange(self.k)] = value

    def extend(self, rows):
        for row in rows:
            self.add(row)
        return self

    @property
    def samples(self):
        return {column: list(values) for column, values in self.reservoirs.items()}

    def __repr__(self):
        return f"ColumnSampler(rows={self.rows}, columns={len(self.reservoirs)}, k={self.k})"
