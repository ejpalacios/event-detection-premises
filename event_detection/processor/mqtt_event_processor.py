import json
import logging
from dataclasses import asdict
from datetime import datetime

import numpy as np
import paho.mqtt.client as mqtt
from pydantic import BaseModel

from event_detection.detectors.event import Event

from .event_processor import EventProcessor

LOGGER = logging.getLogger(__name__)


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.ndarray)):
        return obj.tolist()
    raise TypeError("Type %s not serializable" % type(obj))


class MQTTEventProcessorConfig(BaseModel):
    host: str
    port: int = 1883
    qos: int = 1

    @property
    def output_stream(self) -> EventProcessor:
        return MQTTEventProcessor(host=self.host, port=self.port, qos=self.qos)


class MQTTEventProcessor(EventProcessor):
    def __init__(self, host: str, port: int, qos: int) -> None:
        self._mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._mqtt.connect(host=host, port=port)
        self._qos = qos
        self._mqtt.loop_start()

    def process_event(
        self, detection_time: datetime, device_id: str, event: Event
    ) -> None:
        LOGGER.info(f"Event: {detection_time=}, {device_id=}, {event}")
        self._mqtt.publish(
            f"event/{device_id}",
            json.dumps(asdict(event), default=json_serial),
            qos=self._qos,
        )

    def close(self) -> None:
        self._mqtt.loop_stop()
