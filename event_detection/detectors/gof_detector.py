import logging

import numpy as np
import pandas as pd

from .detector import Detector
from .event import Event
from .stats import goodness_of_fit_event

LOGGER = logging.getLogger(__name__)


class GOFDetector(Detector):
    """
    Event detector based on Jin 2011 paper
    'Robust adaptive event detection in non-intrusive load monitoring for energy aware smart facilities'

    Args:
        event_threshold: the level used to define significant
            appliances, transitions below this level will be ignored.
            Default 20.0
        stat_window: number of samples in the pre and post event windows.
            Default 5
        stat_threshold: alpha threshold for the p-values.
            Default 0.05
    """

    type_1 = "GOF"

    def __init__(
        self,
        event_threshold=20.0,
        stat_window=5,
        stat_threshold=0.05,
        measurements: pd.MultiIndex = pd.MultiIndex.from_tuples([("power", "active")]),
    ):
        self.event_threshold = event_threshold
        self.stat_window = stat_window
        self.stat_threshold = stat_threshold
        super().__init__(measurements, self.stat_window * 2)

    def _init_state(self):
        pass

    @property
    def offset_start_w(self) -> int:
        return self.stat_window

    @property
    def offset_end_w(self) -> int:
        return self.stat_window

    def online_events(self, t_samples, samples) -> tuple[bool, Event]:
        self._check_input_window(t_samples, samples)
        detected = False
        event = None
        time = t_samples[self.stat_window]

        stat_idx = self.stat_window
        pre_event_samples = samples[stat_idx - self.stat_window : stat_idx]
        pos_event_samples = samples[stat_idx : stat_idx + self.stat_window]
        stat, gof, mu_pre, mu_pos = goodness_of_fit_event(
            pre_event=pre_event_samples,
            pos_event=pos_event_samples,
            event_threshold=self.event_threshold,
        )
        if (gof < self.stat_threshold).any():
            detected = True
        event = Event(
            timestamp=time,
            statistic_1_value=gof,
            statistic_1_type=self.type_1,
            pre_event_mean=mu_pre,
            pos_event_mean=mu_pos,
            pre_event_median=np.median(pre_event_samples),
            pos_event_median=np.median(pos_event_samples),
        )
        return detected, event
