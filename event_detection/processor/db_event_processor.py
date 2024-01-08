import logging
from datetime import datetime
from typing import Optional

import psycopg2
from pydantic import BaseModel

from event_detection.detectors.event import Event

from .event_processor import EventProcessor

LOGGER = logging.getLogger(__name__)


class DBEventProcessorConfig(BaseModel):
    host: str
    port: int = 5432
    database: str = "premises"
    user: str = "postgres"
    password: str = "password"

    @property
    def output_stream(self) -> EventProcessor:
        return DBEventProcessor(
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}?sslmode=allow"
        )


class DBEventProcessor(EventProcessor):
    _conn: Optional[psycopg2.extensions.connection] = None

    def __init__(self, connection_uri: Optional[str] = None) -> None:
        if DBEventProcessor._conn is None:
            if connection_uri:
                DBEventProcessor._conn = psycopg2.connect(dsn=connection_uri)
                self.create_tables()
            else:
                raise ValueError("Connection URI has not been set yet")

    @classmethod
    def close(cls) -> None:
        if cls._conn is not None:
            cls._conn.close()
            cls._conn = None

    @classmethod
    def create_tables(cls) -> None:
        cls.create_event_table()

    @classmethod
    def create_event_table(cls) -> None:
        query_create_table = """
            CREATE TABLE IF NOT EXISTS event (
                time TIMESTAMPTZ NOT NULL,
                device_id VARCHAR(20) NOT NULL,
                method VARCHAR(20) NOT NULL,
                detection_time TIMESTAMPTZ,
                statistic_1 REAL,
                statistic_2 REAL,
                pre_state REAL,
                post_state REAL,
                transient REAL,
                PRIMARY KEY (time, device_id, method)
            );
        """
        query_create_hypertable = (
            "SELECT create_hypertable('event', 'time', if_not_exists => TRUE);"
        )

        if cls._conn is not None:
            con: psycopg2.extensions.connection = cls._conn
        with con:
            with con.cursor() as cur:
                cur.execute(query_create_table)
                cur.execute(query_create_hypertable)

    @classmethod
    def process_event(
        cls, detection_time: datetime, device_id: str, event: Event
    ) -> None:
        LOGGER.info(f"Event: {detection_time=}, {device_id=}, {event}")
        query_insert = """
            INSERT INTO event (time, device_id, method, detection_time,
            statistic_1, statistic_2, pre_state, post_state, transient) VALUES 
            (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        stat_2 = None
        if event.statistic_2_value is not None:
            stat_2 = float(event.statistic_2_value[0])
        query_values = (
            event.timestamp,
            device_id,
            f"{event.statistic_1_type}-{event.statistic_2_type}",
            detection_time,
            float(event.statistic_1_value[0]),
            stat_2,
            float(event.pre_event_mean[0]),
            float(event.pos_event_mean[0]),
            float(event.pos_event_mean[0] - event.pre_event_mean[0]),
        )

        if cls._conn is not None:
            con: psycopg2.extensions.connection = cls._conn
        with con:
            try:
                with con.cursor() as cur:
                    cur.execute(query_insert, query_values)
            except psycopg2.Error as e:
                LOGGER.warning(f"Failed to insert rows.\n ERROR: {e}")
