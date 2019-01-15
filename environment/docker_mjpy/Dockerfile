FROM ubuntu:18.04

# Install python and utils
RUN apt-get update && apt-get install -y python3-pip curl unzip \
    libosmesa-dev libglew-dev patchelf libglfw3-dev

# Download mujoco
RUN curl https://www.roboti.us/download/mjpro150_linux.zip --output /tmp/mujoco.zip && \
    mkdir -p /root/.mujoco && \
    unzip /tmp/mujoco.zip -d /root/.mujoco && \
    rm -f /tmp/mujoco.zip

# Set library load path
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin

# Copy the mujoco license key
COPY mjkey.txt /root/.mujoco/mjkey.txt

# Install gym with mujoco
RUN pip3 install gym[mujoco]

# Test the python packages
RUN python3 -c "import gym; gym.make('Humanoid-v2')"
# RUN ln -s python3 /usr/bin/python

RUN pip3 install glfw>=1.4.0
RUN pip3 install numpy>=1.11
RUN pip3 install Cython>=0.27.2
RUN pip3 install imageio>=2.1.2
RUN pip3 install cffi>=1.10
RUN pip3 install imagehash>=3.4
RUN pip3 install Pillow>=4.0.0
RUN pip3 install pycparser>=2.17.0
RUN pip3 install pytest>=3.0.5
RUN pip3 install pytest-instafail==0.3.0
RUN pip3 install scipy>=0.18.0
RUN pip3 install sphinx
RUN pip3 install sphinx_rtd_theme
RUN pip3 install numpydoc
RUN pip3 install cloudpickle==0.5.2
RUN pip3 install cached-property==1.3.1
RUN pip3 install gym==0.10.5
RUN pip3 install gitpython==2.1.7
RUN pip3 install gtimer==1.0.0b5
RUN pip3 install awscli==1.11.179
RUN pip3 install boto3==1.4.8
RUN pip3 install ray==0.2.2
RUN pip3 install path.py==10.3.1
# RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torch
RUN pip3 install torchvision
RUN pip3 install joblib==0.9.4
RUN pip3 install opencv-python==3.4.0.12
RUN pip3 install torchvision==0.2.0
RUN pip3 install sk-video==1.1.10
RUN pip3 install matplotlib

RUN apt-get install -y python3-dev
RUN apt-get install -y build-essential
RUN apt-get install -y libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev

ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

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

COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

WORKDIR /mujoco_py
# Copy over just requirements.txt at first. That way, the Docker cache doesn't
# expire until we actually change the requirements.
COPY ./mujoco-py/requirements.txt /mujoco_py/
COPY ./mujoco-py/requirements.dev.txt /mujoco_py/
RUN pip3 install -r requirements.txt
RUN pip3 install -r requirements.dev.txt

RUN apt-get install -y python3-setuptools

# Delay moving in the entire code until the very end.
ENTRYPOINT ["/mujoco_py/vendor/Xdummy-entrypoint"]
CMD ["pytest"]
COPY ./mujoco-py /mujoco_py
RUN python3 setup.py install

WORKDIR /root/gym_fetch_stack_multiworld
ADD ./gym_fetch_stack_multiworld .
RUN pip3 install -e .

WORKDIR /root/
RUN rm /usr/bin/python
RUN ln -s python3 /usr/bin/python