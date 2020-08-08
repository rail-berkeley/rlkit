# Change these things
CODE_DIRS_TO_MOUNT = [
    '/home/ashvin/code/rlkit',
    # '/home/ashvin/code/doodad',
    '/home/ashvin/code/multiworld',
]
DIR_AND_MOUNT_POINT_MAPPINGS = [
    dict(
        local_dir='/home/ashvin/.mujoco/',
        mount_point='/root/.mujoco',
    ),
]
LOCAL_LOG_DIR = '/home/ashvin/data/s3doodad'
RUN_DOODAD_EXPERIMENT_SCRIPT_PATH = (
    '/home/ashvin/code/rlkit/scripts/run_experiment_from_doodad.py'
)

AWS_S3_PATH="s3://s3doodad/doodad/logs"

# You probably don't need to change things below
# Specifically, the docker image is looked up on dockerhub.com.
DOODAD_DOCKER_IMAGE = 'anair17/railrl-gpu-v7a' # 'anair17/railrl-gpu-v6' # 'vitchyr/railrl-vitchyr-gpu-v2-2'
# DOODAD_DOCKER_IMAGE = "anair17/railrl-gpu-v3"
INSTANCE_TYPE = 'c4.large'
SPOT_PRICE = 0.1
SPOT_PRICE_LOOKUP = {'c4.large': 0.1, 'm4.large': 0.1, 'm4.xlarge': 0.2, 'm4.2xlarge': 0.4}

# GPU_DOODAD_DOCKER_IMAGE = "anair17/railrl-gpu-v3"
GPU_DOODAD_DOCKER_IMAGE = 'anair17/railrl-gpu-v7a' #  # 'vitchyr/railrl-vitchyr-gpu-v3'
GPU_INSTANCE_TYPE = 'g3.4xlarge'
GPU_SPOT_PRICE = 1.0
GPU_AWS_IMAGE_ID = "ami-0680f279"

GPU_SINGULARITY_IMAGE = "/home/ashvin/gpuv6cuda9.img"
SINGULARITY_IMAGE = "/home/ashvin/gpuv6cuda9.img"
SINGULARITY_PRE_CMDS = [
    'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin'
]

SSH_HOSTS=dict(
    localhost=dict(
        username="ashvin",
        hostname="localhost",
    ),
    surgical1=dict(
        username="ashvin",
        hostname="surgical1",
    ),
    newton1=dict(
        username="ashvin",
        hostname="newton1",
    ),
    newton2=dict(
        username="ashvin",
        hostname="newton2",
    ),
    newton3=dict(
        username="ashvin",
        hostname="newton3",
    ),
    newton4=dict(
        username="ashvin",
        hostname="newton4",
    ),
    newton5=dict(
        username="ashvin",
        hostname="newton5",
    ),
    newton6=dict(
        username="ashvin",
        hostname="newton6",
    ),
    newton7=dict(
        username="ashvin",
        hostname="newton7",
    ),
)
SSH_DEFAULT_HOST="fail"
# SSH_LOG_DIR = '/home/ashvin/data/s3doodad'
SSH_LOG_DIR = '/media/4tb/ashvin/data/s3doodad'
SSH_PRIVATE_KEY = '/home/ashvin/.ssh/id_rsa'

REGION_TO_GPU_AWS_IMAGE_ID = {
    'us-west-1': "ami-0b2985bdb79796529",
    'us-east-1': "ami-0680f279",
    'us-west-2': "ami-0e57f21d309963e66",
    'us-east-2': "ami-0c323cf1a03b771e5",
}

REGION_TO_GPU_AWS_AVAIL_ZONE = {}

# This really shouldn't matter and in theory could be whatever
OUTPUT_DIR_FOR_DOODAD_TARGET = '/tmp/doodad-output/'
