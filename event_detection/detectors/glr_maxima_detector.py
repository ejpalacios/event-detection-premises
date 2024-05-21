import logging
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from .detector import Detector
from .event import Event
from .stats import log_likelihood_ratio_event

LOGGER = logging.getLogger(__name__)


class GLRMaximaDetector(Detector):
    """
    Event detector based on Pereira 2017 paper
    'Developing and evaluating a probabilistic event detector for non-intrusive load monitoring'

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

    type_1 = "GLR"
    type_2 = "MAXIMA"

    def __init__(
        self,
        event_threshold: float = 20.0,
        stat_window: tuple[int, int] = (5, 5),
        stat_threshold: float = 150.0,
        range_std: Optional[tuple[float, float]] = None,
        maxima_precision: int = 5,
        measurements: pd.MultiIndex = pd.MultiIndex.from_tuples([("power", "active")]),
    ):
        self.event_threshold = event_threshold
        self.pre_window, self.pos_window = stat_window
        self.stat_threshold = stat_threshold
        self.maxima_precision = maxima_precision
        self.event_window = 2 * self.maxima_precision + 1
        self.range_std = range_std
        super().__init__(
            measurements,
            np.maximum(self.event_window, self.pre_window + 1) + self.pos_window,
        )

    def _init_state(self):
        self.stat_w = deque(
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
        return np.maximum(self.event_window - 1, self.pre_window)

    @property
    def offset_end_w(self) -> int:
        return self.pos_window + 1

    def online_events(self, t_samples, samples) -> tuple[bool, Event]:
        self._check_input_window(t_samples, samples)
        detected = False
        event = None
        vote_time = t_samples[self.maxima_precision]

        # Calculate GLR for last point in voting window

        stat_idx = self.total_length_w - self.pos_window - 1
        measurement = np.array([samples[stat_idx]])
        pre_event_data = samples[stat_idx - self.pre_window : stat_idx]
        pos_event_data = samples[stat_idx + 1 : stat_idx + self.pos_window + 1]
        ll_ratio, mu_pre, mu_pos = log_likelihood_ratio_event(
            samples=measurement,
            pre_event=pre_event_data,
            pos_event=pos_event_data,
            event_threshold=self.event_threshold,
            range_std=self.range_std,
        )

        self.stat_w.append(ll_ratio)

        # Keep track of the means before and after
        self.mean_pre_w.append(mu_pre)
        self.mean_pos_w.append(mu_pos)
        # Keep track of the medians before and after
        self.median_pre_w.append(np.median(pre_event_data))
        self.median_pos_w.append(np.median(pos_event_data))

        # Calculate the maximum point and check if it is in the middle
        maxima_idx = int(np.argmax(np.array(list(self.stat_w)), axis=0))
        if maxima_idx == self.maxima_precision:
            detected = True

        event = Event(
            timestamp=vote_time,
            statistic_1_value=self.stat_w[self.maxima_precision],
            statistic_1_type=self.type_1,
            pre_event_mean=self.mean_pre_w[self.maxima_precision],
            pos_event_mean=self.mean_pos_w[self.maxima_precision],
            pre_event_median=self.median_pre_w[self.maxima_precision],
            pos_event_median=self.median_pos_w[self.maxima_precision],
        )
        return detected, event
