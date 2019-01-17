FROM denden047/mujoco-py:latest

# Copy the mujoco license key
COPY mjkey.txt /root/.mujoco/mjkey.txt

# Install gym with mujoco
RUN python -c "import mujoco_py"
RUN pip install gym[mujoco]

RUN pip install glfw>=1.4.0
RUN pip install numpy>=1.11
RUN pip install Cython>=0.27.2
RUN pip install imageio>=2.1.2
RUN pip install cffi>=1.10
RUN pip install imagehash>=3.4
RUN pip install Pillow>=4.0.0
RUN pip install pycparser>=2.17.0
RUN pip install pytest>=3.0.5
RUN pip install pytest-instafail==0.3.0
RUN pip install scipy>=0.18.0
RUN pip install sphinx
RUN pip install sphinx_rtd_theme
RUN pip install numpydoc
RUN pip install cloudpickle==0.5.2
RUN pip install cached-property==1.3.1
RUN pip install gym==0.10.5
RUN pip install gitpython==2.1.7
RUN pip install gtimer==1.0.0b5
RUN pip install awscli==1.11.179
RUN pip install boto3==1.4.8
RUN pip install ray==0.2.2
RUN pip install path.py==10.3.1
RUN pip install torchvision
RUN pip install joblib==0.9.4
RUN pip install opencv-python==3.4.0.12
RUN pip install torchvision==0.2.0
RUN pip install sk-video==1.1.10
RUN pip install matplotlib

RUN pip install torch

ARG CACHEBUST=1 

WORKDIR /root/gym_fetch_stack_multiworld
ADD ./gym_fetch_stack_multiworld .
RUN pip install -e .

RUN python -c "import mujoco_py"