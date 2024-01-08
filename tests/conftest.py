import numpy as np
import pytest


@pytest.fixture(scope="module")
def norm_low():
    return {"mu": 10.0, "std": 1.0}


@pytest.fixture(scope="module")
def norm_high():
    return {"mu": 100.0, "std": 1.0}


@pytest.fixture(scope="module")
def samples_low(norm_low):
    samples = np.random.normal(norm_low["mu"], norm_low["std"], (100, 1))
    return samples


@pytest.fixture(scope="module")
def samples_high(norm_high):
    samples = np.random.normal(norm_high["mu"], norm_high["std"], (100, 1))
    return samples


@pytest.fixture(scope="module")
def up_event(samples_low, samples_high):
    return np.concatenate((samples_low, samples_high))


@pytest.fixture(scope="module")
def down_event(samples_low, samples_high):
    return np.concatenate((samples_high, samples_low))


@pytest.fixture(scope="module")
def pulse_event(samples_low, samples_high):
    return np.concatenate((samples_low, samples_high, samples_low))


@pytest.fixture(scope="module")
def no_event(samples_low):
    return np.concatenate((samples_low, samples_low))
