import numpy as np
import pandas as pd
import pytest

from event_detection.detectors import (  # GOFDetector,
    GLRDetector,
    GLRVoteDetector,
    GLRCUMVoteDetector,
    GOFVoteDetector,
    HartDetector,
)
from event_detection.detectors.detector import Detector

detectors_instances: list[Detector] = [
    HartDetector(),
    GLRDetector(range_std=(1.0, np.inf)),
    GLRVoteDetector(range_std=(1.0, np.inf)),
    GLRCUMVoteDetector(range_std=(1.0, np.inf)),
    # GOFDetector(),  # Many false positives for low-frequency data
    GOFVoteDetector(),
]

test_cases = []
id_cases = []
for detector_instance in detectors_instances:
    for dimension in range(0, 2):
        test_cases.append((detector_instance, dimension + 1))
        id_cases.append(
            f"{detector_instance.type_1}-{detector_instance.type_2}-{dimension+1}"
        )


@pytest.mark.parametrize("detector, dimensions", test_cases, ids=id_cases)
def test_up_event(norm_low, norm_high, up_event, detector, dimensions):
    up_event_n = np.repeat(up_event, dimensions, axis=1)
    ref_event = [norm_high["mu"] - norm_low["mu"]] * dimensions
    ref_state = [norm_high["mu"]] * dimensions

    df = pd.DataFrame(up_event_n)
    df.index = pd.to_datetime(df.index, unit="s")
    events = detector.offline_events(df)
    transitions = events.transitions
    states = events.pos_states
    n_events = len(transitions.index.values)
    assert n_events == 1, f"Should have detected 1 event, got {n_events}"
    assert transitions.index.values[0] == np.datetime64(100, "s")
    assert transitions.values[0] == pytest.approx(ref_event, 1.0)
    assert states.values[0] == pytest.approx(ref_state, 1.0)


@pytest.mark.parametrize("detector, dimensions", test_cases, ids=id_cases)
def test_down_event(norm_low, norm_high, down_event, detector, dimensions):
    down_event_n = np.repeat(down_event, dimensions, axis=1)
    ref_event = [norm_low["mu"] - norm_high["mu"]] * dimensions
    ref_state = [norm_low["mu"]] * dimensions

    df = pd.DataFrame(down_event_n)
    df.index = pd.to_datetime(df.index, unit="s")
    events = detector.offline_events(df)
    transitions = events.transitions
    states = events.pos_states
    n_events = len(transitions.index.values)
    assert n_events == 1, f"Should have detected 1 event, got {n_events}"
    assert transitions.index.values[0] == np.datetime64(100, "s")
    assert transitions.values[0] == pytest.approx(ref_event, 1.0)
    assert states.values[0] == pytest.approx(ref_state, 1.0)


@pytest.mark.parametrize("detector, dimensions", test_cases, ids=id_cases)
def test_pulse_event(norm_low, norm_high, pulse_event, detector, dimensions):
    pulse_event_n = np.repeat(pulse_event, dimensions, axis=1)
    ref_event_up = [norm_high["mu"] - norm_low["mu"]] * dimensions
    ref_state_up = [norm_high["mu"]] * dimensions
    ref_event_down = [norm_low["mu"] - norm_high["mu"]] * dimensions
    ref_state_down = [norm_low["mu"]] * dimensions

    df = pd.DataFrame(pulse_event_n)
    df.index = pd.to_datetime(df.index, unit="s")
    events = detector.offline_events(df)
    transitions = events.transitions
    states = events.pos_states
    n_events = len(transitions.index.values)
    assert n_events == 2, f"Should have detected 2 events, got {n_events}"
    assert transitions.index.values[0] == np.datetime64(100, "s")
    assert transitions.index.values[1] == np.datetime64(200, "s")
    assert transitions.values[0] == pytest.approx(ref_event_up, 1.0)
    assert states.values[0] == pytest.approx(ref_state_up, 1.0)
    assert transitions.values[1] == pytest.approx(ref_event_down, 1.0)
    assert states.values[1] == pytest.approx(ref_state_down, 1.0)


@pytest.mark.parametrize("detector, dimensions", test_cases, ids=id_cases)
def test_no_event(no_event, detector, dimensions):
    no_event_n = np.repeat(no_event, dimensions, axis=1)

    events = detector.offline_events(pd.DataFrame(no_event_n))
    transitions = events.transitions
    n_events = len(transitions.index.values)
    assert n_events == 0, f"Should have detected 0 events, got {n_events}"
