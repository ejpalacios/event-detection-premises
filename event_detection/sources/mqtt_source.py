from paho.mqtt.client import Client
from pydantic import BaseModel


class MQTTSourceConfig(BaseModel):
    host: str
    port: int = 1883
    qos: int = 1

    @property
    def input_stream(self) -> Client:
        reader = Client()
        reader.connect(host=self.host, port=self.port)
        return reader
