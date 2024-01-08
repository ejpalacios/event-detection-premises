from abc import ABC, abstractmethod, abstractproperty

import numpy as np
import pandas as pd

from .event import Event, EventBuffer


class Detector(ABC):
    """Abstract detector class

    Args:
        measurements: list of measurements used in the detector
        total_length_w: total expected length of the detection window
    """

    type_1 = ""
    type_2 = ""

    def __init__(self, measurements: pd.Index, total_length_w: int) -> None:
        self._measurements = measurements
        self._n_measurements = len(self._measurements)
        self.total_length_w = total_length_w
        self._event_buffer = EventBuffer(self._measurements)
        self.reset()

    def reset(self) -> None:
        """Clean events and states buffers and initialise internal attributes"""
        self._empty_buffers()
        self._init_state()

    def _empty_buffers(self) -> None:
        """Empty the events and state buffers"""
        self._event_buffer.clean_buffer(self._measurements)

    @abstractmethod
    def _init_state(self) -> None:
        """Reset the internal state of the class attributes"""

    @property
    def measurements(self) -> pd.Index:
        """List of measurements used in the detector"""
        return self._measurements

    @measurements.setter
    def measurements(self, measurements: pd.Index) -> None:
        self._measurements = measurements
        self._n_measurements = len(measurements)
        self.reset()

    @property
    def n_measurements(self) -> int:
        """Number of measurements"""
        return self._n_measurements

    @abstractproperty
    def offset_start_w(self) -> int:
        pass

    @abstractproperty
    def offset_end_w(self) -> int:
        pass

    def _check_input_window(self, t_samples: np.ndarray, samples: np.ndarray) -> None:
        """Checks that the online event detection window length is correct

        Args:
            t_samples: indexes of the sample windows as datetime
            samples: samples of one or more signals where the event will be
                detected.

        Raises:
            ValueError: the online detection window length is incorrect
        """
        n_samples = len(samples)
        n_t_samples = len(t_samples)
        if n_samples != n_t_samples:
            raise ValueError(
                f"The lenghts of `t_samples` and `samples` must be the same {n_t_samples} != {n_samples}"
            )
        if len(samples) != self.total_length_w:
            raise ValueError(
                f"The window length should be equal to {self.total_length_w}, got {n_samples}"
            )

    def offline_events(self, dataframe: pd.DataFrame, detailed=False) -> EventBuffer:
        """Tags events in a historical timeseries

        Args:
            dataframe: pandas dataframe with historical measurements.
            It must include a datetime index and one or more value columns
            detailed: returns full event buffer

        Returns:
            np.ndarray: array of values calculated by the detection algorithm
        """
        self.measurements = dataframe.columns

        indexes = dataframe.index.values
        values = dataframe.values
        n_samples = values.shape[0]

        for i in range(self.offset_start_w, n_samples - self.offset_end_w):
            index = indexes[i - self.offset_start_w : i + self.offset_end_w]
            sample = values[i - self.offset_start_w : i + self.offset_end_w]
            detected, event = self.online_events(index, sample)
            if detailed:
                self._event_buffer.add_event(event.timestamp, event)
            else:
                if detected:
                    self._event_buffer.add_event(event.timestamp, event)

        return self._event_buffer

    @abstractmethod
    def online_events(
        self, t_samples: np.ndarray, samples: np.ndarray
    ) -> tuple[bool, Event]:
        """Tags events online using a detection window

        Args:
            t_samples: indexes of the sample windows as datetime
            samples: samples of one or more signals where the event will be
                detected.

        Returns:
            np.ndarray: vector of values calculated by the detection algorithm
        """
