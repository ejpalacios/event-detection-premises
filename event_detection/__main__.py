import argparse
import logging
import os

import yaml

from event_detection.service import DetectorService, DetectorServiceConfig

logging.basicConfig(level=os.getenv("LOG_LEVEL", "DEBUG"))
LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event detection service")
    parser.add_argument(
        "--config",
        help="Path to YAML configuration file",
    )

    args = parser.parse_args()
    config = None
    if args.config:
        with open(args.config) as f:
            config = DetectorServiceConfig(**yaml.load(f, Loader=yaml.FullLoader))
    else:
        config = DetectorServiceConfig()  # type: ignore

    LOGGER.info(f"Running service {config.detector.method}")
    DetectorService(
        config.device_id, config.detector.build(), config.client, config.output_streams
    ).run()
