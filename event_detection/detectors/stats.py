import logging
from typing import Optional

import numpy as np
from scipy.stats import chi2, norm

LOGGER = logging.getLogger(__name__)


def online_mean(sample: np.ndarray, previous_mean: np.ndarray, n: int) -> np.ndarray:
    """Calculates the online mean using Welford's algorithm

    Args:
        sample: current observation
        previous_mean: last calculated mean value
        n: number of observations including the current one

    Returns:
        float: updated online mean value
    """
    return previous_mean + (sample - previous_mean) / n


def online_sum_squares(
    sample: np.ndarray,
    previous_sum_squares: np.ndarray,
    previous_mean: np.ndarray,
    current_mean: np.ndarray,
) -> np.ndarray:
    """Calculates the online sum of squares using Welford's algorithm

    Args:
        sample: current observation
        previous_sum_squares: last calculated sum of squares
        previous_mean: last calculated mean value
        current_mean: mean value for the current observation

    Returns:
        float: updated online sum of squares
    """
    return previous_sum_squares + (sample - previous_mean) * (sample - current_mean)


def norm_mle(
    samples: np.ndarray, std_range: Optional[tuple[float, float]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Maximum likelihood estimation of normal distribution

    Arg:
        samples: observations of the random variable
        std_range: range to limit the standard deviation (min,max)

    Returns:
        tuple[float, float]: estimated mean and standard deviation
    """
    mu = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    if std_range is not None:
        std = np.minimum(np.maximum(std, std_range[0]), std_range[1])
    return mu, std


def norm_log_likelihood_ratio(
    samples: np.ndarray,
    mu_pre: np.ndarray,
    mu_pos: np.ndarray,
    std_pre: np.ndarray,
    std_pos: np.ndarray,
) -> np.ndarray:
    """Log likelihood ratio of two normal distributions

    Args:
        samples: observations of the random variable
        mu_pre: mean of the pre-event window
        std_pre: standard deviation of the pre-event window
        mu_post: mean of the post-event window
        std_post: standard deviation of the post-event window

    Returns:
        float: log likelihood ratio
    """
    ll_post = norm.logpdf(samples, loc=mu_pos, scale=std_pos)
    ll_pre = norm.logpdf(samples, loc=mu_pre, scale=std_pre)
    ll_ratio = np.sum(ll_post - ll_pre, axis=0)
    return ll_ratio


def log_likelihood_ratio_event(
    samples: np.ndarray,
    pos_event: np.ndarray,
    pre_event: Optional[np.ndarray] = None,
    pre_stats: Optional[tuple[np.ndarray, np.ndarray]] = None,
    event_threshold: float = 20,
    range_std: Optional[tuple[float, float]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log likelihood ratio for of an event

    Args:
        samples: observations over which the log likelihood is calulcated
        pos_event: post-event observations Used to estimate the post-event distribution
        pre_event: pre-event observations. Used to estimate the pre-event distribution
        pre_stats: pre-event statistics. Must be provided if pre_event is not given
        event_threshold: minimum change detection event_threshold
        range_std: dynamic range of the standard deviation

    Returns:
        tuple[float, float, float]: GLR ratio, mu pre-event, mu post-event
    """
    n_measurements = np.array(pos_event).shape[1]
    ll_ratio = np.zeros(n_measurements)
    if pre_event is not None:
        mu_pre, std_pre = norm_mle(pre_event, range_std)
    elif pre_stats is not None:
        mu_pre = pre_stats[0]
        std_pre = pre_stats[1]
    else:
        raise ValueError(
            "Either the pre_event window or its statistics (mean, std) must be provided"
        )

    mu_pos, std_pos = norm_mle(pos_event, range_std)

    mu_delta = mu_pos - mu_pre
    if (np.abs(mu_delta) > event_threshold).any():
        ll_ratio = norm_log_likelihood_ratio(samples, mu_pre, mu_pos, std_pre, std_pos)
    return ll_ratio, mu_pre, mu_pos


def goodness_of_fit_event(
    pre_event: np.ndarray,
    pos_event: np.ndarray,
    event_threshold: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chi-squared goodness of fit test of an event

    Args:
        pre_event: pre-event observations
        pos_event: post-event observations
        event_threshold: minimum change detection event_threshold

    Returns:
        tuple[float, float, float]: chi-2 statistics, p-value, mu pre-vent,
            mu post-event
    """
    n_measurements = np.array(pos_event).shape[1]
    df = np.array(pos_event).shape[0] - 1
    p_value = np.ones(n_measurements)
    statistic = np.zeros(n_measurements)
    mu_pre = np.mean(pre_event, axis=0)
    mu_pos = np.mean(pos_event, axis=0)
    state_delta = mu_pos - mu_pre
    # pos_event_norm = np.sum(pre_event) / np.sum(pos_event) * pos_event
    if (np.abs(state_delta) > event_threshold).any():
        statistic = np.sum((pre_event - pos_event) ** 2 / pos_event, axis=0)
        p_value = chi2.sf(statistic, df)
    return statistic, p_value, mu_pre, mu_pos
