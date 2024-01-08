import logging
from typing import Optional

import numpy as np
import pandas as pd

from .detector import Detector
from .event import Event
from .stats import log_likelihood_ratio_event

LOGGER = logging.getLogger(__name__)


class GLRDetector(Detector):
    """Event detector based on Luo 2002 paper

    Args:
        event_threshold: the level used to define significant appliances,
            transitions below this level will be ignored. Default 20.0
        stat_window: number of samples in the pre- and post-event window.
            Default (5, 5)
        stat_threshold: threshold of the GLR statistic for event detection.
            Default 150.0
        range_std: range of the standard deviation.
            Default None
    """

    type_1 = "GLR"

    def __init__(
        self,
        event_threshold: float = 20.0,
        stat_window: tuple[int, int] = (5, 5),
        stat_threshold: float = 150.0,
        range_std: Optional[tuple[float, float]] = None,
        measurements: pd.MultiIndex = pd.MultiIndex.from_tuples([("power", "active")]),
    ):
        self.event_threshold = event_threshold
        self.pre_window, self.pos_window = stat_window
        self.stat_threshold = stat_threshold
        self.range_std = range_std
        super().__init__(measurements, self.pre_window + self.pos_window + 1)

    def _init_state(self):
        pass

    @property
    def offset_start_w(self) -> int:
        return self.pre_window

    @property
    def offset_end_w(self) -> int:
        return self.pos_window + 1

    def online_events(self, t_samples, samples) -> tuple[bool, Event]:
        self._check_input_window(t_samples, samples)
        detected = False
        event = None

        time = t_samples[self.pre_window]
        measurement = np.array([samples[self.pre_window]])
        pre_event_samples = samples[0 : self.pre_window]
        pos_event_samples = samples[
            self.pre_window + 1 : self.pre_window + self.pos_window + 1
        ]
        ll_ratio, mu_pre, mu_pos = log_likelihood_ratio_event(
            samples=measurement,
            pre_event=pre_event_samples,
            pos_event=pos_event_samples,
            event_threshold=self.event_threshold,
            range_std=self.range_std,
        )
        if (ll_ratio > self.stat_threshold).any():
            detected = True
        event = Event(
            timestamp=time,
            statistic_1_value=ll_ratio,
            statistic_1_type=self.type_1,
            pre_event_mean=mu_pre,
            pos_event_mean=mu_pos,
        )

        return detected, event
