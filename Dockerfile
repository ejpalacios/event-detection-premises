FROM python:3.9-slim AS base

WORKDIR /code

COPY requirements.txt .
RUN pip3 install -r requirements.txt

FROM base AS premises_nilm
COPY event_detection event_detection
CMD [ "python", "-m", "event_detection"]
