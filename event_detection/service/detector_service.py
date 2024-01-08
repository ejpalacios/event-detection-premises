import json
import logging
import time as ts
from collections import deque
from datetime import datetime
from typing import Any, Optional

import numpy as np
from paho.mqtt.client import Client, MQTTMessage
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from scipy.ndimage import median_filter

from event_detection.detectors import (
    GLRDetector,
    GLRVoteDetector,
    GOFDetector,
    GOFVoteDetector,
    HartDetector,
)
from event_detection.detectors.detector import Detector
from event_detection.processor import DBEventProcessorConfig, MQTTEventProcessorConfig
from event_detection.processor.event_processor import EventProcessor
from event_detection.sources import MQTTSourceConfig

from .service import Service

LOGGER = logging.getLogger(__name__)


class DetectorConfig(BaseModel):
    method: str
    event_threshold: float = 20.0
    state_threshold: float = 20.0
    min_n_samples: int = 2
    pre_window: int = 5
    pos_window: int = 5
    stat_window: int = 5
    stat_threshold: float = 150.0
    min_std: Optional[float] = None
    max_std: Optional[float] = None
    event_window: int = 20
    vote_threshold: int = 15

    def build(self) -> Detector:
        detector_intance: Optional[Detector] = None
        if self.method == "HART":
            detector_intance = HartDetector(
                self.event_threshold, self.state_threshold, self.min_n_samples
            )
        elif self.method == "GLR":
            range_std = None
            if self.min_std is not None and self.max_std is not None:
                range_std = (self.min_std, self.max_std)
            detector_intance = GLRDetector(
                event_threshold=self.event_threshold,
                stat_window=(self.pre_window, self.pos_window),
                stat_threshold=self.stat_threshold,
                range_std=range_std,
            )
        elif self.method == "GLR_VOTING":
            if self.min_std is not None and self.max_std is not None:
                range_std = (self.min_std, self.max_std)
            detector_intance = GLRVoteDetector(
                event_threshold=self.event_threshold,
                stat_window=(self.pre_window, self.pos_window),
                event_window=self.event_window,
                vote_threshold=self.vote_threshold,
                stat_threshold=self.stat_threshold,
                range_std=range_std,
            )
        elif self.method == "GOF":
            detector_intance = GOFDetector(
                event_threshold=self.event_threshold,
                stat_window=self.stat_window,
                stat_threshold=self.stat_threshold,
            )
        elif self.method == "GOF_VOTING":
            detector_intance = GOFVoteDetector(
                event_threshold=self.event_threshold,
                stat_window=self.stat_window,
                stat_threshold=self.stat_threshold,
                event_window=self.event_window,
                vote_threshold=self.vote_threshold,
            )
        else:
            raise ValueError(f"{self.method} is not a valid detection algorithm")
        return detector_intance


class DetectorServiceConfig(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    device_id: str
    client: MQTTSourceConfig
    detector: DetectorConfig
    mqtt: Optional[MQTTEventProcessorConfig] = None
    db: Optional[DBEventProcessorConfig] = None

    @property
    def output_streams(self) -> list[EventProcessor]:
        outputs = []
        if self.mqtt is not None:
            outputs.append(self.mqtt.output_stream)
        if self.db is not None:
            outputs.append(self.db.output_stream)
        return outputs


class DetectorService(Service):
    def __init__(
        self,
        device_id: str,
        detector: Detector,
        client: MQTTSourceConfig,
        output_streams: list[EventProcessor],
        n_filter=2,
    ) -> None:
        self.device_id = device_id
        self.detector = detector
        self.client = client
        self.mqtt = client.input_stream
        self.output_streams = output_streams
        self.n_filter = n_filter
        self.n_total = (
            detector.offset_start_w + detector.offset_end_w + (self.n_filter * 2)
        )

        self.value_buffer = deque(np.zeros((self.n_total, 1)), maxlen=self.n_total)
        self.time_buffer = deque(np.zeros(self.n_total), maxlen=self.n_total)

        self.n_samples = 0
        self.detector.reset()

    def sub_telegram(self, client: Client, userdata: Any, msg: MQTTMessage) -> None:
        data = json.loads(msg.payload)
        t_sample = datetime.fromisoformat(data["P1_MESSAGE_TIMESTAMP"]["value"])
        sample = data["CURRENT_ELECTRICITY_USAGE"]["value"] * 1000
        # Take the topic since EQUIPMENT_IDENTIFIER is incorrect for now on dsmr-parser
        # encoded_id = data["EQUIPMENT_IDENTIFIER"]["value"]
        # device_id = bytearray.fromhex(encoded_id).decode()

        ## Process event
        self.time_buffer.append(t_sample)
        self.value_buffer.append(np.array([sample]))
        if self.n_samples >= self.n_total:
            ft_value_buffer = median_filter(
                np.array(list(self.value_buffer)), size=self.n_filter * 2 + 1
            )

            time = np.array(list(self.time_buffer)[self.n_filter : -self.n_filter])
            ft_value = np.array(ft_value_buffer[self.n_filter : -self.n_filter])

            t_s = ts.perf_counter()
            detected, event = self.detector.online_events(time, ft_value)
            t_e = ts.perf_counter()
            LOGGER.debug(f"{t_sample.isoformat()}, {t_e - t_s}")
            if detected:
                for output_stream in self.output_streams:
                    output_stream.process_event(t_sample, self.device_id, event)
        else:
            self.n_samples += 1

    def run(self) -> None:
        self.n_samples = 0
        self.detector.reset()
        self.mqtt.subscribe(topic=f"telegram/{self.device_id}", qos=1)
        self.mqtt.message_callback_add(
            sub=f"telegram/{self.device_id}", callback=self.sub_telegram
        )
        try:
            self.mqtt.loop_forever()
        except KeyboardInterrupt:
            self.mqtt.loop_stop()
            for output_stream in self.output_streams:
                output_stream.close()
