---
layout: post

title: Build a Machine Learning Docker image

tip-number: 36
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: Learn how to build lightweight docker images for Machine Learning
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - more
---

Docker is good for packaging binaries for production use as well as for developpement. One way to build the docker image is to have one `Dockerfile` and install all you binaries dependencies in it but this may lead to large docker images. For instance, build tools (e.g. `gcc`) and related artifacts are usually added during the compilation of the binaries but they may not be needed at runtime and having them is a non necessary overhead.

We can leaverage [Docker multi-stage builds](https://docs.docker.com/develop/develop-images/multistage-build/) to create a clean final Docker image for the runtime phase.

1. have a builder image to compile and install all dependencies
```Dockerfile
FROM continuumio/miniconda3:latest as builder

COPY requirements.txt /requirements.txt

RUN . /root/.bashrc && \
    conda init bash && \
    yes | conda create -n py38 python=3.8 && \
    conda activate py38 && \
    conda config --set auto_activate_base false && \
    yes | conda install -c conda-forge --file requirements.txt || true; \
    conda clean -afy
```

You can declare the dependencies in a `requirements.txt` file, something like this
```yaml
pandas
numpy
gcc
prophet
```

2. have the final image that copies the binaries and propoerly configure the `PATH` to have access to them
```Dockerfile
FROM busybox

COPY --from=builder /opt/conda/envs/py38 /opt/conda/envs/py38

RUN echo 'alias python=/opt/conda/envs/py38/bin/python' >> ~/.bashrc && . /root/.bashrc
```

Note that the target runtime image can be anything, a lightweight image like `busybox`, a python image or even a java (e.g `openjdk:11`).

3. The final `Dockerfile` will combine the two build steps and would look like this
```Dockerfile
# Stage 1: Builder
FROM continuumio/miniconda3:latest as builder

COPY requirements.txt /requirements.txt

RUN . /root/.bashrc && \
    conda init bash && \
    yes | conda create -n py38 python=3.8 && \
    conda activate py38 && \
    conda config --set auto_activate_base false && \
    yes | conda install -c conda-forge --file requirements.txt || true; \
    conda clean -afy

# Stage 2: Runtime
FROM busybox

COPY --from=builder /opt/conda/envs/py38 /opt/conda/envs/py38

RUN echo 'alias python=/opt/conda/envs/py38/bin/python' >> ~/.bashrc && . /root/.bashrc
```

