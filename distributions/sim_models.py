import abc
import numpy as np
import scipy.linalg as lin
import scipy.special as spl
import scipy.stats as stats
import math
from joblib import Parallel, delayed


# from memory_profiler import profile
dtype = np.float32


class _distribution:
    __metaclass__ = abc.ABCMeta

    def __init__(self, x_dim=0, rng=np.random):
        self.rng = rng.default_rng()
        self.x_dim = x_dim

    @abc.abstractmethod
    def sim(self):
        """Method to generate sample from the distribution"""
        return

    def entropy(self):
        """Method returning the entropy of the given distribution. Use default method if entropy unknown.
        Defaults to returning none."""
        return None

    def pdf(self, x):
        """PDF of the distribution"""
        return 0

    def logpdf(self, x):
        """log(pdf) of the distribution. Useful for entropy calculations"""
        return 0


class Uniform(_distribution):
    def __init__(self, low, high, N=1):
        super().__init__(x_dim=N)
        self.low = low
        self.high = high

    def sim(self, n_samples=1000):
        return (self.high - self.low) * self.rng.random(
            (n_samples, self.x_dim), dtype=dtype
        ) + self.low

    def entropy(self):
        return self.x_dim * np.log(self.high - self.low)


class Gaussian(_distribution):
    def __init__(self, mu, sigma2, N=1):
        super().__init__(x_dim=N)
        # make mu,sigma vectors if input is a scalar
        if np.isscalar(mu):
            mu = [mu] * N
            sigma2 = sigma2 * np.eye(N)
        self.mu = np.asarray(mu, dtype=dtype)
        self.sigma = np.asarray(sigma2, dtype=dtype)

    def sim(self, n_samples=1000):
        return self.rng.multivariate_normal(mean=self.mu, cov=self.sigma, size=n_samples).astype(
            dtype
        )

    def entropy(self):
        return 0.5 * np.log(lin.det(self.sigma)) + self.x_dim / 2 * (1 + np.log(2 * np.pi))


class Limited_Gaussian(Gaussian):
    def __init__(self, mu, sigma2, limit, N=1):
        super().__init__(mu, sigma2, N)
        self.limit = limit

    def sim(self, n_samples=1000):
        samples = super().sim(n_samples)
        mask = np.linalg.norm(samples, axis=1) <= self.limit
        return samples[mask, :]

    def entropy(self):
        # Found through MAF uniformization. Does not scale linearly with dimensions
        if self.x_dim == 3 and self.sigma[0, 0] == 1 and self.mu[0] == 0:
            return 4.106
        else:
            return None


class Exponential(_distribution):
    """Class to simulate an exponential distribution
    https://en.wikipedia.org/wiki/Exponential_distribution
    """

    def __init__(self, lamb):
        super().__init__(x_dim=1)
        self.lamb = lamb

    def sim(self, n_samples):
        return self.rng.exponential(scale=1 / self.lamb, size=(n_samples, 1)).astype(dtype)

    def entropy(self):
        return 1 - np.log(self.lamb)


class Exponential_sum(_distribution):
    """Class to simulate the sum of two independent exponential variables
    https://en.wikipedia.org/wiki/Exponential_distribution
    """

    def __init__(self, lambda1, lambda2):
        super().__init__(x_dim=1)
        assert lambda1 > lambda2, "lambda1 > lambda2 for this entropy test"
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.expo1 = Exponential(lambda1)
        self.expo2 = Exponential(lambda2)

    def sim(self, n_samples):
        return self.expo1.sim(n_samples) + self.expo2.sim(n_samples)

    def entropy(self):
        return (
            1
            + np.euler_gamma
            + np.log((self.lambda1 - self.lambda2) / (self.lambda1 * self.lambda2))
            + spl.digamma(self.lambda1 / (self.lambda1 - self.lambda2))
        )


class Laplace(_distribution):
    """Simulate the laplace distribution
    https://en.wikipedia.org/wiki/Laplace_distribution"""

    def __init__(self, mu, b, N=1):
        super().__init__(x_dim=N)
        self.mu = np.asarray(mu, dtype=dtype)
        self.b = np.asarray(b, dtype=dtype)

    def sim(self, n_samples):
        return self.rng.laplace(self.mu, self.b, (n_samples, self.x_dim)).astype(dtype)

    def entropy(self):
        return (1 + np.log(2 * self.b)) * self.x_dim

    def pdf(self, x):
        return 1 / (2 * self.b) * np.exp(-np.abs(x - self.mu) / self.b)

    def logpdf(self, x):
        return -self.x_dim * np.log(2 * self.b) - np.sum(np.abs(x - self.mu) / self.b, axis=1)


class Cauchy(_distribution):
    def __init__(self, mu, gamma, N=1):
        super().__init__(x_dim=N)
        self.mu = mu
        self.gamma = gamma

    def sim(self, n_samples):
        return stats.cauchy.rvs(loc=self.mu, scale=self.gamma, size=(n_samples, self.x_dim)).astype(
            dtype
        )

    def entropy(self):
        return np.log(4 * np.pi * self.gamma) * self.x_dim


class Logistic(_distribution):
    def __init__(self, mu, s, N=1):
        super().__init__(x_dim=N)
        self.s = s
        self.mu = mu

    def sim(self, n_samples):
        return stats.logistic.rvs(loc=self.mu, scale=self.s, size=(n_samples, self.x_dim)).astype(
            dtype
        )

    def entropy(self):
        return (np.log(self.s) + 2) * self.x_dim
