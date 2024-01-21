# Dockerfile for training
# Start from a base Python image
FROM python:3.10

ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}
ENV PYTHONUNBUFFERED=0

WORKDIR /app

# Copy data folder to Docker
COPY ../data /app/data

# Copy models folder to Docker
COPY ../models /app/models
COPY ../training /app/training

# Copy the code itself
COPY inference /app/inference

# Copy usefull scripts and configs
COPY ../utils.py /app
COPY ../${CONF_PATH} /app

# Install any necessary packages listed in requirements.txt
COPY ../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt