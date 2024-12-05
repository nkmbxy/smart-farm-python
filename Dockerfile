FROM python:3.9-slim-buster

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends libgl1-mesa-dev libglib2.0-0 && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r /code/requirements.txt && \
    rm -rf /var/lib/apt/lists/*

COPY . /code

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]