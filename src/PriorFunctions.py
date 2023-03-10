import numpy as np


def simple_prior():
    bcf = np.random.uniform(low=0, high=1)
    pi = np.random.uniform(low=0, high=1)
    return np.array([bcf, pi])


def prior_DV():
    DV = np.random.uniform(low=0, high=1)
    bcf = np.random.uniform(low=0, high=1)
    return np.array([DV, bcf])
