import logging
from collections import deque

import numpy as np
import pandas as pd

from .detector import Detector
from .event import Event
from .stats import goodness_of_fit_event

LOGGER = logging.getLogger(__name__)


class GOFVoteDetector(Detector):
    """
    Event detector based on De Baets 2017 paper
    'On the Bayesian optimization and robustness of event detection methods in NILM'

    Args:
        event_threshold: the level used to define significant
            appliances, transitions below this level will be ignored.
            Default 20.0
        stat_window: number of samples in the pre and post event windows.
            Default 5
        stat_threshold: alpha threshold for the p-values.
            Default 0.05
        event_window: number of samples in event detection window.
            Default 20
        vote_threshold: threshold of votes for event detection.
            Default 15
    """

    type_1 = "GOF"
    type_2 = "VOTING"

    def __init__(
        self,
        event_threshold=20.0,
        stat_window=5,
        stat_threshold=0.05,
        event_window=20,
        vote_threshold=15,
        measurements: pd.MultiIndex = pd.MultiIndex.from_tuples([("power", "active")]),
    ):
        self.event_threshold = event_threshold
        self.stat_window = stat_window
        self.stat_threshold = stat_threshold
        self.event_window = event_window
        self.vote_threshold = vote_threshold
        super().__init__(
            measurements,
            np.maximum(self.event_window, self.stat_window + 1) + self.stat_window - 1,
        )

    def _init_state(self):
        self.stat_w = deque(
            np.zeros((self.event_window, self.n_measurements)), maxlen=self.event_window
        )
        self.votes_w = deque(
            np.zeros((self.event_window, self.n_measurements)), maxlen=self.event_window
        )
        self.mean_pre_w = deque(
            np.zeros((self.event_window, self.n_measurements)), maxlen=self.event_window
        )
        self.mean_pos_w = deque(
            np.zeros((self.event_window, self.n_measurements)), maxlen=self.event_window
        )
        self.median_pre_w = deque(
            np.zeros((self.event_window, self.n_measurements)), maxlen=self.event_window
        )
        self.median_pos_w = deque(
            np.zeros((self.event_window, self.n_measurements)), maxlen=self.event_window
        )

    @property
    def offset_start_w(self) -> int:
        return np.maximum(self.event_window - 1, self.stat_window)

    @property
    def offset_end_w(self) -> int:
        return self.stat_window

    def online_events(self, t_samples, samples) -> tuple[bool, Event]:
        self._check_input_window(t_samples, samples)
        detected = False
        event = None

        # Calculate GOF for last point in voting window

        stat_idx = self.total_length_w - self.stat_window
        pre_event_samples = samples[stat_idx - self.stat_window : stat_idx]
        pos_event_samples = samples[stat_idx : stat_idx + self.stat_window]
        stat, gof, mu_pre, mu_pos = goodness_of_fit_event(
            pre_event=pre_event_samples,
            pos_event=pos_event_samples,
            event_threshold=self.event_threshold,
        )

        # Only appen stat if significant
        if (gof < self.stat_threshold).any():
            self.stat_w.append(stat)
        else:
            self.stat_w.append(np.zeros((self.n_measurements)))

        # Keep track of the means before and after
        self.mean_pre_w.append(mu_pre)
        self.mean_pos_w.append(mu_pos)
        # Keep track of the means before and after
        self.median_pre_w.append(np.median(pre_event_samples))
        self.median_pos_w.append(np.median(pos_event_samples))

        # Calculate the point that gets a vote
        self.votes_w.append(np.zeros((self.n_measurements), dtype=int))
        vote_idx = np.argmax(np.array(list(self.stat_w)), axis=0)
        for i, vote in enumerate(vote_idx):
            if self.stat_w[vote][i] > 0:
                self.votes_w[vote][i] += 1

        if (self.votes_w[0] > self.vote_threshold).any():
            detected = True

        event = Event(
            timestamp=t_samples[stat_idx - self.event_window + 1],
            statistic_1_value=self.stat_w[0],
            statistic_1_type=self.type_1,
            statistic_2_value=self.votes_w[0],
            statistic_2_type=self.type_2,
            pre_event_mean=self.mean_pre_w[0],
            pos_event_mean=self.mean_pos_w[0],
            pre_event_median=self.median_pre_w[0],
            pos_event_median=self.median_pos_w[0],
        )
        return detected, event
