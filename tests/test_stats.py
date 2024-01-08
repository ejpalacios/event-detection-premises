import numpy as np
import pytest

import event_detection.detectors.stats as st


@pytest.mark.parametrize("dimensions", [1, 2])
def test_online_mean(samples_low: np.ndarray, dimensions: int):
    samples_low_n = np.repeat(samples_low, dimensions, axis=1)
    n_measurements = samples_low_n.shape[1]
    online_mean = np.zeros(n_measurements)

    offline_mean = np.mean(samples_low_n, axis=0)

    n = 0
    for sample in samples_low:
        n += 1
        online_mean = st.online_mean(sample, online_mean, n)
    assert (
        len(online_mean) == dimensions
    ), f"Expected {dimensions} dimensions, got {len(online_mean)}"
    assert offline_mean == pytest.approx(
        online_mean, 0.01
    ), f"Online mean should be equal to {offline_mean}~0.01, got {online_mean}"


@pytest.mark.parametrize("dimensions", [1, 2])
def test_online_variance(samples_low: np.ndarray, dimensions: int):
    samples_low_n = np.repeat(samples_low, dimensions, axis=1)
    n_measurements = samples_low_n.shape[1]
    online_mean = np.zeros(n_measurements)
    online_sum_sq = np.zeros(n_measurements)

    offline_var = np.var(samples_low_n, ddof=1, axis=0)

    n = 0
    for sample in samples_low:
        pre_mean = online_mean
        n += 1
        online_mean = st.online_mean(sample, online_mean, n)
        online_sum_sq = st.online_sum_squares(
            sample, online_sum_sq, pre_mean, online_mean
        )
    online_var = online_sum_sq / (n - 1)

    assert (
        len(online_var) == dimensions
    ), f"Expected {dimensions} dimensions, got {len(online_var)}"
    assert offline_var == pytest.approx(
        online_var, 0.01
    ), f"Online variance should be equal to {offline_var}~0.01, got {online_var}"


@pytest.mark.parametrize("dimensions", [1, 2])
def test_norm_mle(norm_low, samples_low, dimensions):
    samples_low_n = np.repeat(samples_low, dimensions, axis=1)
    ref_mu = [norm_low["mu"]] * dimensions
    ref_std = [norm_low["std"]] * dimensions

    mu, std = st.norm_mle(samples_low_n)

    assert len(mu) == dimensions, f"Expected {dimensions} dimensions, got {len(mu)}"
    assert len(std) == dimensions, f"Expected {dimensions} dimensions, got {len(std)}"
    assert mu == pytest.approx(
        ref_mu, 0.5
    ), f"Mean must be equal to {ref_mu}~0.5, got: {mu}"
    assert std == pytest.approx(
        ref_std, 0.5
    ), f"Standard deviation must be equal to {ref_std}~0.5, got: {std}"


@pytest.mark.parametrize("dimensions", [1, 2])
def test_norm_mle_range(norm_low, samples_low, dimensions):
    samples_low_n = np.repeat(samples_low, dimensions, axis=1)
    ref_mu = [norm_low["mu"]] * dimensions
    ref_std_min = [2.0] * dimensions
    ref_std_max = [3.0] * dimensions

    mu_limit, std_limit = st.norm_mle(samples_low_n, std_range=(2.0, 3.0))
    assert (
        len(mu_limit) == dimensions
    ), f"Expected {dimensions} dimensions, got {len(mu_limit)}"
    assert (
        len(std_limit) == dimensions
    ), f"Expected {dimensions} dimensions, got {len(std_limit)}"
    assert mu_limit == pytest.approx(
        ref_mu, 0.5
    ), f"Mean must be equal to {ref_mu}~0.5, got: {mu_limit}"
    assert (
        ref_std_min <= std_limit
    ).all(), f"Standard deviation must be >= 2.0, got: {std_limit}"
    assert (
        ref_std_max >= std_limit
    ).all(), f"Standard deviation must be <= 3.0, got: {std_limit}"


@pytest.mark.parametrize("dimensions", [1, 2])
def test_norm_log_likelihood_ratio(
    norm_low, norm_high, samples_low, samples_high, dimensions
):
    samples_low_n = np.repeat(samples_low, dimensions, axis=1)
    samples_high_n = np.repeat(samples_high, dimensions, axis=1)
    ref_mu_low = [norm_low["mu"]] * dimensions
    ref_mu_high = [norm_high["mu"]] * dimensions
    ref_std_low = [norm_low["std"]] * dimensions
    ref_std_high = [norm_high["std"]] * dimensions

    ll_ratio_a = st.norm_log_likelihood_ratio(
        samples_low_n, ref_mu_low, ref_mu_high, ref_std_low, ref_std_high
    )
    assert (
        len(ll_ratio_a) == dimensions
    ), f"Expected {dimensions} dimensions, got {len(ll_ratio_a)}"
    assert (
        ll_ratio_a < 0.0
    ).all(), f"log-likelihood ratio of samples in the pre-event distribution must be negative, got: {ll_ratio_a}"

    ll_ratio_b = st.norm_log_likelihood_ratio(
        samples_high_n, ref_mu_low, ref_mu_high, ref_std_low, ref_std_high
    )
    assert (
        len(ll_ratio_b) == dimensions
    ), f"Expected {dimensions} dimensions, got {len(ll_ratio_a)}"
    assert (
        ll_ratio_b > 0.0
    ).all(), f"log-likelihood ratio of samples in the post-event distribution must be positive, got: {ll_ratio_b}"

    ll_ratio_same = st.norm_log_likelihood_ratio(
        samples_low_n, ref_mu_low, ref_mu_low, ref_std_low, ref_std_low
    )
    assert (
        len(ll_ratio_same) == dimensions
    ), f"Expected {dimensions} dimensions, got {len(ll_ratio_same)}"
    assert (
        ll_ratio_same == 0.0
    ).all(), f"log-likelihood ratio with similar pre and post distribution must be zero, got: {ll_ratio_same}"


@pytest.mark.parametrize("dimensions", [1, 2])
def test_log_likelihood_ratio_event(up_event, down_event, no_event, dimensions):
    up_event_n = np.repeat(up_event, dimensions, axis=1)
    down_event_n = np.repeat(down_event, dimensions, axis=1)
    no_event_n = np.repeat(no_event, dimensions, axis=1)

    stat_val = 100.0
    ll_up_event, mu_pre, mu_pos = st.log_likelihood_ratio_event(
        samples=[up_event_n[100]],
        pre_event=up_event_n[0:100],
        pos_event=up_event_n[101:],
        event_threshold=0.0,
    )
    assert (
        len(ll_up_event) == dimensions
    ), f"Expected {dimensions} dimensions, got {len(ll_up_event)}"
    assert (
        ll_up_event > stat_val
    ).all(), f"GLR for up event should be greter than {stat_val}, got {ll_up_event}"
    ll_down_event, mu_pre, mu_pos = st.log_likelihood_ratio_event(
        samples=[down_event_n[100]],
        pre_event=down_event_n[0:100],
        pos_event=down_event_n[101:],
        event_threshold=0.0,
    )
    assert (
        len(ll_down_event) == dimensions
    ), f"Expected {dimensions} dimensions, got {len(ll_down_event)}"
    assert (
        ll_down_event > stat_val
    ).all(), f"GLR for down event should be greter than {stat_val}, got {ll_down_event}"
    ll_no_event, mu_pre, mu_pos = st.log_likelihood_ratio_event(
        samples=[no_event_n[100]],
        pre_event=no_event_n[0:100],
        pos_event=no_event_n[101:],
        event_threshold=0.0,
    )
    assert (
        len(ll_no_event) == dimensions
    ), f"Expected {dimensions} dimensions, got {len(ll_no_event)}"
    assert (
        ll_no_event < stat_val
    ).all(), f"GLR for no event should be lower than {stat_val}, got {ll_no_event}"


@pytest.mark.parametrize("dimensions", [1, 2])
def test_log_likelihood_ratio_event_pre_computed(up_event, dimensions):
    up_event_n = np.repeat(up_event, dimensions, axis=1)

    stat_val = 100.0
    mu_pre = np.mean(up_event_n[0:100], axis=0)
    std_pre = np.std(up_event_n[0:100], axis=0)
    ll_up_event_cal, mu_pre_cal, mu_pos_cal = st.log_likelihood_ratio_event(
        samples=[up_event_n[100]],
        pre_stats=(mu_pre, std_pre),
        pos_event=up_event[101:],
        event_threshold=0.0,
    )
    assert (
        mu_pre == mu_pre_cal
    ).all(), f"Precomputed mean should be equal to estimate {mu_pre_cal}, got {mu_pre}"
    assert (
        ll_up_event_cal > stat_val
    ).all(), f"GLR for up event should be greter than {stat_val}, got {ll_up_event_cal}"

    ll_up_event, mu_pre, mu_pos = st.log_likelihood_ratio_event(
        samples=[up_event_n[100]],
        pre_event=up_event_n[0:100],
        pos_event=up_event_n[101:],
        event_threshold=0.0,
    )
    assert (
        ll_up_event_cal == ll_up_event
    ).all(), f"GLR for precomputed event should be equatal than estimated {ll_up_event}, got {ll_up_event_cal}"


@pytest.mark.parametrize("dimensions", [1, 2])
def test_log_likelihood_ratio_event_no_input(up_event, dimensions):
    up_event_n = np.repeat(up_event, dimensions, axis=1)

    with pytest.raises(ValueError):
        ll_up_event, mu_pre, mu_pos = st.log_likelihood_ratio_event(
            samples=[up_event_n[100]],
            pos_event=up_event_n[101:],
            event_threshold=0.0,
        )


@pytest.mark.parametrize("dimensions", [1, 2])
def test_goodness_of_fit_event(up_event, down_event, no_event, dimensions):
    up_event_n = np.repeat(up_event, dimensions, axis=1)
    down_event_n = np.repeat(down_event, dimensions, axis=1)
    no_event_n = np.repeat(no_event, dimensions, axis=1)

    confident_interval = 0.05
    _, gof_up_event, mu_pre, mu_pos = st.goodness_of_fit_event(
        pre_event=up_event_n[0:100],
        pos_event=up_event_n[100:],
        event_threshold=0.0,
    )
    assert (
        len(gof_up_event) == dimensions
    ), f"Expected {dimensions} dimensions, got {len(gof_up_event)}"
    assert (
        gof_up_event < confident_interval
    ).all(), f"GOF for up event should be lower than {confident_interval}, got {gof_up_event}"
    _, gof_down_event, mu_pre, mu_pos = st.goodness_of_fit_event(
        pre_event=down_event_n[0:100],
        pos_event=down_event_n[100:],
        event_threshold=0.0,
    )
    assert (
        len(gof_down_event) == dimensions
    ), f"Expected {dimensions} dimensions, got {len(gof_down_event)}"
    assert (
        gof_down_event < confident_interval
    ).all(), f"GOF for down event should be lower than {confident_interval}, got {gof_down_event}"
    _, gof_no_event, mu_pre, mu_pos = st.goodness_of_fit_event(
        pre_event=no_event_n[0:100],
        pos_event=no_event_n[100:],
        event_threshold=0.0,
    )
    assert (
        len(gof_no_event) == dimensions
    ), f"Expected {dimensions} dimensions, got {len(gof_no_event)}"
    assert (
        gof_no_event > confident_interval
    ).all(), f"GOF for no event should be higher than {confident_interval}, got {gof_no_event}"
