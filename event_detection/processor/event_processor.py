from abc import abstractmethod
from datetime import datetime

from event_detection.detectors.event import Event


class EventProcessor:
    @abstractmethod
    def process_event(
        self, detection_time: datetime, device_id: str, event: Event
    ) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
