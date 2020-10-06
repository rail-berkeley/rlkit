FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3.7-dev python3.7 python3-pip
RUN virtualenv --python=python3.7 env

RUN rm /usr/bin/python
RUN ln -s /env/bin/python3.7 /usr/bin/python
RUN ln -s /env/bin/pip3.7 /usr/bin/pip
RUN ln -s /env/bin/pytest /usr/bin/pytest

RUN apt-get update -q \
&& DEBIAN_FRONTEND=noninteractive apt-get install -y \
libxcursor-dev \
libxrandr-dev \
libxinerama-dev \
libxi-dev \
mesa-common-dev \
zip \
unzip \
make \
g++ \
python2.7 \
wget \
vulkan-utils \
mesa-vulkan-drivers \
libopenmpi-dev \
libspatialindex-dev \
python3-dev \
python3-tk \
libpython3.7 \
libcairo2-dev \
python3-cairo-dev

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
 && rm -rf /var/lib/apt/lists/*

 # install python essentials
RUN pip install -q --no-cache-dir -U pip
RUN pip install -q --no-cache-dir -U setuptools

# install tensorflow
RUN pip install -q --no-cache-dir ipython requests numpy pandas quandl
RUN pip install -q tensorflow-gpu==1.13.1

# dependencies for carbongym
# Note. python 2.7 is only for packman.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 libxcursor-dev \
 libxrandr-dev \
 libxinerama-dev \
 libxi-dev \
 mesa-common-dev \
 zip \
 unzip \
 make \
 g++ \
 python2.7 \
 wget \
 vulkan-utils \
 mesa-vulkan-drivers \
 libopenmpi-dev

# python dependencies for carbongym
RUN pip install -q \
numpy-quaternion click mpi4py joblib gym

# python dependencies for carbongym-utils
RUN pip install -q \
autolab_core autolab_perception visualization simple_zmq numba numpy-quaternion gym

RUN pip install -q torch torchvision

RUN pip install -q gtimer

RUN pip install -q \
boto3 boto cloudpickle awscli

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git

RUN mkdir /home/test
RUN cd /home/test && git clone https://github.com/NVIDIA/apex.git
RUN cd /home/test/apex/ && pip install -q -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./