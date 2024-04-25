import paho.mqtt.client as mqtt
from pydantic import BaseModel


class MQTTSourceConfig(BaseModel):
    host: str
    port: int = 1883
    qos: int = 1

    @property
    def input_stream(self) -> mqtt.Client:
        reader = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        reader.connect(host=self.host, port=self.port)
        return reader
