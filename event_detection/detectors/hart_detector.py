import logging
from typing import Optional

import numpy as np
import pandas as pd

from .detector import Detector
from .event import Event, EventBuffer
from .stats import online_mean

LOGGER = logging.getLogger(__name__)


class HartDetector(Detector):
    """Event detector based on Hart 1985 paper

    Implements and expert heuristics approach to event detection

    Args:
        event_threshold: the level used to define significant
            appliances, events below this level will be ignored.
        state_threshold: maximum difference between highest and lowest
            value within a state.
        min_n_samples: number of samples to consider constituting a
            new state.
    """

    type_1 = "HART"

    def __init__(
        self,
        event_threshold: float = 20.0,
        state_threshold: float = 20.0,
        min_n_samples: int = 2,
        measurements: pd.MultiIndex = pd.MultiIndex.from_tuples([("power", "active")]),
    ):
        # Configuration parameters
        self.event_threshold = event_threshold
        self.state_threshold = state_threshold
        self.min_n_samples = min_n_samples
        super().__init__(measurements, 2)

    def _init_state(self) -> None:
        self.N = 0  # N stores the number of samples in state
        self.est_state = np.zeros(self.n_measurements)  # Current state estimate
        self.last_t_event: Optional[np.datetime64] = None
        self.last_state = np.zeros(self.n_measurements)  # Previous state estimate
        self.ongoing_change = False  # Power change in progress

    @property
    def offset_start_w(self) -> int:
        return 1

    @property
    def offset_end_w(self) -> int:
        return 1

    def offline_events(self, dataframe: pd.DataFrame, detailed=False) -> EventBuffer:
        super().offline_events(dataframe, detailed)
        # Appending last edge
        last_event_power = self.est_state - self.last_state
        if (
            np.abs(last_event_power) > self.event_threshold
        ).any() and self.last_t_event is not None:
            event = Event(
                timestamp=self.last_t_event,
                statistic_1_value=np.ones(self.n_measurements),
                statistic_1_type=self.type_1,
                pre_event_mean=self.last_state,
                pos_event_mean=self.est_state,
                pre_event_median=self.last_state,
                pos_event_median=self.est_state,
            )
            if event is not None:
                self._event_buffer.add_event(event.timestamp, event)
        return self._event_buffer

    def online_events(
        self, t_samples: np.ndarray, samples: np.ndarray
    ) -> tuple[bool, Event]:
        self._check_input_window(t_samples, samples)
        # Step 2: Check if power delta is over the state limit
        detected = False

        time = t_samples[1]

        event = Event(
            timestamp=time,
            statistic_1_value=np.zeros(self.n_measurements),
            statistic_1_type=self.type_1,
            pre_event_mean=self.last_state,
            pos_event_mean=self.est_state,
            pre_event_median=self.last_state,
            pos_event_median=self.est_state,
        )
        measurement = samples[1]
        previous_measurement = samples[0]
        instantaneous_change = False  # power changing this second

        delta_measurement = np.abs(measurement - previous_measurement)
        if (delta_measurement > self.state_threshold).any():
            instantaneous_change = True

        # Step 3: Identify if an event is just starting, if so, process it
        if instantaneous_change and (not self.ongoing_change):
            LOGGER.debug(
                f"Starting event {instantaneous_change=}, {delta_measurement=}"
            )

            # 3A. Calculate event size
            if self.last_t_event is not None:
                last_event_power = self.est_state - self.last_state
                if (
                    np.abs(last_event_power) > self.event_threshold
                ).any() and self.N > self.min_n_samples:
                    # 3A. Send time and magnitue to event buffer (ignore first)
                    detected = True
                    event.timestamp = self.last_t_event
                    event.statistic_1_value = np.ones(self.n_measurements)
                # 3B
                self.last_state = self.est_state
                # 3C
                self.last_t_event = time
            else:
                self.last_state = self.est_state
                # 3C
                self.last_t_event = time

        # Step 4: if a new state is starting, zero counter
        if instantaneous_change:
            self.N = 0

        # Step 6: increment counter (need to be done first due to the algorithm for online mean)
        self.N += 1
        # Step 5: update our estimate for state's power
        self.est_state = online_mean(measurement, self.est_state, self.N)

        # Step 7
        self.ongoing_change = instantaneous_change
        return detected, event
