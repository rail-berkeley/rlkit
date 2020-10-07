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
    python3-cairo-dev \
    build-essential \
    curl \
    git \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
    apt-get clean

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

#RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
#RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3.7-dev python3.7 python3-pip
#RUN virtualenv --python=python3.7 env
#
#RUN rm /usr/bin/python
#RUN ln -s /env/bin/python3.7 /usr/bin/python
#RUN ln -s /env/bin/pip3.7 /usr/bin/pip
#RUN ln -s /env/bin/pytest /usr/bin/pytest
#
## install python essentials
#RUN pip install -q --no-cache-dir -U pip
#RUN pip install -q --no-cache-dir -U setuptools
#RUN pip install -q --no-cache-dir ipython requests numpy pandas quandl ipdb
#
## python dependencies for carbongym
#RUN pip install -q \
#numpy-quaternion click mpi4py joblib gym
#
## python dependencies for carbongym-utils
#RUN pip install -q \
#autolab_core autolab_perception visualization simple_zmq numba numpy-quaternion gym
#
#RUN pip install -q torch torchvision
#
#RUN pip install -q gtimer
#
#RUN pip install -q \
#boto3 boto cloudpickle awscli
#
#RUN mkdir /home/test
#RUN cd /home/test && git clone https://github.com/NVIDIA/apex.git
#RUN cd /home/test/apex/ && pip install -q -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
#
#RUN mkdir /home/test/carbongym
#RUN cd /home/test/carbongym && git init . && git pull https://824d1ec21bc8cd8f10eeabb09cae73bc42f0f0cb@github.com/mihdalal/carbongym.git
#RUN cd /home/test/carbongym/python && pip install -e .
