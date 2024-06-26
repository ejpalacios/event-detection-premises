from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass()
class Event:
    """Event object"""

    # np.datetime64: Event timestamp
    timestamp: np.datetime64

    # np.ndarray: Mean values of post-event window
    pre_event_mean: np.ndarray

    # np.ndarray: Mean values of pre-event window
    pos_event_mean: np.ndarray

    # np.ndarray: Median values of post-event window
    pre_event_median: np.ndarray

    # np.ndarray: Median values of pre-event window
    pos_event_median: np.ndarray

    # str: Type of event metric
    statistic_1_type: str

    # np.ndarray: Values of the event metric
    statistic_1_value: np.ndarray

    # str: Type of secondary event metric
    statistic_2_type: Optional[str] = field(default=None)

    # np.ndarray: Values of the secondary event metric
    statistic_2_value: Optional[np.ndarray] = field(default=None)

    # object: Event features for training
    features: Optional[object] = field(default=None)

    @property
    # np.ndarray: Delta between post-event and pre-event means
    def delta_mean(self) -> np.ndarray:
        """np.ndarray: Different between mean post- and pre-event windows"""
        return self.pos_event_mean - self.pre_event_mean

    @property
    # np.ndarray: Delta between post-event and pre-event medians
    def delta_median(self) -> np.ndarray:
        """np.ndarray: Different between mean post- and pre-event windows"""
        return self.pos_event_median - self.pre_event_median


class EventBuffer:
    def __init__(self, measurements: pd.Index) -> None:
        self._measurements = measurements
        self._event_buffer: dict[np.datetime64, Event] = dict()

    def clean_buffer(self, measurements: pd.Index) -> None:
        self._measurements = measurements
        self._event_buffer.clear()

    def add_event(self, timestamp: np.datetime64, event: Event):
        self._event_buffer[timestamp] = event

    def delete_event(self, timestamp: np.datetime64) -> Optional[Event]:
        return self._event_buffer.pop(timestamp)

    def _time_index(self) -> pd.DatetimeIndex:
        count = len(self._event_buffer)
        indexes = np.fromiter(
            self._event_buffer.keys(), dtype="datetime64[ns]", count=count
        )
        return pd.to_datetime(indexes, utc=True)

    def _build_df(self, data) -> pd.DataFrame:
        return pd.DataFrame(
            index=self._time_index(), data=data, columns=self._measurements
        )

    def events(self) -> dict[np.datetime64, Event]:
        return self._event_buffer

    @property
    def delta_mean(self) -> pd.DataFrame:
        events = [value.delta_mean for value in self._event_buffer.values()]
        return self._build_df(events)

    @property
    def delta_median(self) -> pd.DataFrame:
        events = [value.delta_median for value in self._event_buffer.values()]
        return self._build_df(events)

    @property
    def pos_event_mean(self) -> pd.DataFrame:
        events = [value.pos_event_mean for value in self._event_buffer.values()]
        return self._build_df(events)

    @property
    def pre_event_mean(self) -> pd.DataFrame:
        events = [value.pre_event_mean for value in self._event_buffer.values()]
        return self._build_df(events)

    @property
    def pos_event_median(self) -> pd.DataFrame:
        events = [value.pos_event_median for value in self._event_buffer.values()]
        return self._build_df(events)

    @property
    def pre_event_median(self) -> pd.DataFrame:
        events = [value.pre_event_median for value in self._event_buffer.values()]
        return self._build_df(events)

    @property
    def statistic_1(self) -> pd.DataFrame:
        events = [value.statistic_1_value for value in self._event_buffer.values()]
        return self._build_df(events)

    @property
    def statistic_2(self) -> pd.DataFrame:
        events = [value.statistic_2_value for value in self._event_buffer.values()]
        return self._build_df(events)

    def extract_event_profiles(
        self,
        agg_df: pd.DataFrame,
        window: tuple[int, int],
        filter=lambda x: True,
        norm=True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract event transition profile from main dataframe

        Args:
            agg_df (pd.DataFrame): Power dataframe
            window (tuple[int,int]): Pre and post event window to capture
            filter (callable): Filter to apply to delta_mean values
            norm (boolean): Normalise event to minimum value
        """
        profiles = []
        timestamps = []
        for idx, value in enumerate(self._event_buffer.values()):
            if filter(value.delta_mean):
                timestamp_start = pd.to_datetime(
                    value.timestamp - np.timedelta64(window[0], "s"), utc=True
                )
                timestamp_end = pd.to_datetime(
                    value.timestamp + np.timedelta64(window[1], "s"), utc=True
                )
                event = agg_df.loc[timestamp_start:timestamp_end]  # type: ignore
                if len(event) == window[0] + window[1] + 1:
                    event_out = event - event.min() if norm else event
                    profiles.append(event_out.values)
                    timestamps.append(value.timestamp)
        return np.array(timestamps), np.array(profiles)
