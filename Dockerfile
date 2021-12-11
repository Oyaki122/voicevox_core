# TIPS
# ====
# Build:
#     docker build -t voicevox_core .
# Run:
#     docker run -it voicevox_core bash

FROM python:3.9.6-slim AS build-env

# Install requirements with apt
RUN apt-get update -yqq \
    &&  apt-get install -yqq \
    curl \
    tar \
    unzip \
    cmake \
    g++ \
    git \
    libsndfile-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy EntryPoint
COPY ./ /voicevox_core/.

# Setup libtorch
RUN curl -sLO https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-linux-x64-1.10.0.tgz
RUN ls -d onnxruntime*.tgz | xargs -n1 -i bash -c 'tar -xzvf {}; rm {}'
RUN mkdir -p voicevox_core/onnxruntime && mv /onnxruntime*/* /voicevox_core/onnxruntime && rm -r /onnxruntime*

WORKDIR /voicevox_core

RUN pip install -U pip && pip install -q cython numpy

