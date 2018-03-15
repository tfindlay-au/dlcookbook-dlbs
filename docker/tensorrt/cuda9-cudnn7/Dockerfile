FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER sergey.serebryakov@hpe.com

ARG version=nv-tensorrt-repo-ubuntu1604-ga-cuda9.0-trt3.0.4-20180208_1-1_amd64.deb
ENV TENSORRT_PACKAGE=$version

COPY $version /

RUN dpkg -i /${TENSORRT_PACKAGE} && \
    rm /${TENSORRT_PACKAGE} && \
    apt-get update && apt-get install -y --no-install-recommends \
        numactl \
        build-essential \
        cmake \
        git \
        wget \
        tensorrt \
        python-libnvinfer-doc \
        uff-converter-tf \
        libboost-program-options-dev \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*


COPY /tensorrt /tensorrt

RUN cd /tensorrt && rm -rf -- ./build && \
    mkdir build && cd ./build/ && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/tensorrt .. && \
    make && make install && \
    cd / && rm -rf /tensorrt


ENV PATH /opt/tensorrt/bin:$PATH
