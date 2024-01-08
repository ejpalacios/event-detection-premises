from abc import ABC, abstractmethod


class Service(ABC):
    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError
