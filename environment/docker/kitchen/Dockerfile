FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04


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

RUN apt-get update && apt-get install -y zsh libglfw3-dev libgles2-mesa-dev patchelf

RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3.7-dev python3.7 python3-pip
RUN virtualenv --python=python3.7 env

RUN rm /usr/bin/python
RUN ln -s /env/bin/python3.7 /usr/bin/python
RUN ln -s /env/bin/pip3.7 /usr/bin/pip
RUN ln -s /env/bin/pytest /usr/bin/pytest
RUN pip install --upgrade pip

ENV LANG C.UTF-8

RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
COPY ./mjkey.txt /root/.mujoco/
RUN cp -R /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200/
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN mkdir /root/code
WORKDIR /root/code

RUN pip install mujoco_py
WORKDIR /mujoco_py

# For atari-py
RUN apt-get install -y zlib1g-dev swig cmake

RUN pip install glfw>=1.4.0
RUN pip install numpy>=1.11
RUN pip install Cython>=0.27.2
RUN pip install imageio>=2.1.2
RUN pip install cffi>=1.10
RUN pip install imagehash>=3.4
RUN pip install ipdb
RUN pip install Pillow>=4.0.0
RUN pip install pycparser>=2.17.0
RUN pip install pytest>=3.0.5
RUN pip install pytest-instafail
RUN pip install scipy>=0.18.0
RUN pip install sphinx
RUN pip install sphinx_rtd_theme
RUN pip install numpydoc
RUN pip install cloudpickle==1.3.0
RUN pip install cached-property==1.3.1
RUN pip install gitpython==2.1.7
RUN pip install gtimer
RUN pip install awscli
RUN pip install boto3
RUN pip install joblib
RUN pip install sk-video
RUN pip install torch torchvision
RUN pip install opencv-python==4.3.0.36
RUN pip install matplotlib
RUN pip install stable_baselines3
RUN pip install h5py
RUN pip install termcolor
RUN pip install click
RUN pip install git+git://github.com/deepmind/dm_control@master#egg=dm_control
RUN pip install git+git://github.com/aravindr93/mjrl@master#egg=mjrl
RUN pip install numba
RUN pip install numpy-quaternion
RUN mkdir /usr/lib/nvidia-000/
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/lib/nvidia-000
RUN python -c "import mujoco_py"
