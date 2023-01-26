import json
import os.path as osp

import numpy as np


class Ecdf():
    def __init__(self, cdf=None, values=None, fname=None):
        self._cdf = cdf
        self._values = values
        if fname is not None and osp.exists(fname):
            with open(fname, 'r') as f:
                dict = json.load(f)
                self._cdf = np.array(dict['cdf'])
                self._values = np.array(dict['values'])

    def fit(self, data, nBins=50):
        #  https://alpynepyano.github.io/healthyNumerics/posts/sampling_arbitrary_distributions_with_python.html
        n, bins, = np.histogram(data, bins=nBins)
        bin_width = bins[1] - bins[0]
        self._values = bins[0:-1] + 0.5*bin_width
        self._pdf = n / np.sum(n)
        self._cdf = np.zeros_like(bins)
        self._cdf[1:] = np.cumsum(self._pdf)

        return None

    def save(self, fname):
        dict={'cdf':list(self._cdf), 'values':list(self._values)}
        with open(fname, 'w') as f:
            json.dump(dict, f)

    # this could be better https://stackoverflow.com/questions/3209362/how-to-plot-empirical-cdf-ecdf
    def get_sampled_element(self):
        a = np.random.uniform(0, 1)
        return np.argmax(self._cdf >= a) - 1

    def pdf(self, bins, *args):
        assert len(bins) == len(self._pdf) + 1
        return self._pdf

    def rvs(self, size):
        x = np.zeros(size)
        for i in range(size):
            x[i] = self._values[self.get_sampled_element()]
        return x
