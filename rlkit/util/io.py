import joblib
import numpy as np
import pickle

import boto3

from rlkit.launchers.conf import LOCAL_LOG_DIR, AWS_S3_PATH
import os

PICKLE = 'pickle'
NUMPY = 'numpy'
JOBLIB = 'joblib'


def local_path_from_s3_or_local_path(filename):
    relative_filename = os.path.join(LOCAL_LOG_DIR, filename)
    if os.path.isfile(filename):
        return filename
    elif os.path.isfile(relative_filename):
        return relative_filename
    else:
        return sync_down(filename)


def sync_down(path, check_exists=True):
    is_docker = os.path.isfile("/.dockerenv")
    if is_docker:
        local_path = "/tmp/%s" % (path)
    else:
        local_path = "%s/%s" % (LOCAL_LOG_DIR, path)

    if check_exists and os.path.isfile(local_path):
        return local_path

    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    if is_docker:
        from doodad.ec2.autoconfig import AUTOCONFIG
        os.environ["AWS_ACCESS_KEY_ID"] = AUTOCONFIG.aws_access_key()
        os.environ["AWS_SECRET_ACCESS_KEY"] = AUTOCONFIG.aws_access_secret()

    full_s3_path = os.path.join(AWS_S3_PATH, path)
    bucket_name, bucket_relative_path = split_s3_full_path(full_s3_path)
    try:
        bucket = boto3.resource('s3').Bucket(bucket_name)
        bucket.download_file(bucket_relative_path, local_path)
    except Exception as e:
        local_path = None
        print("Failed to sync! path: ", path)
        print("Exception: ", e)
    return local_path


def sync_down_folder(path):
    is_docker = os.path.isfile("/.dockerenv")
    if is_docker:
        local_path = "/tmp/%s" % (path)
    else:
        local_path = "%s/%s" % (LOCAL_LOG_DIR, path)

    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    if is_docker:
        from doodad.ec2.autoconfig import AUTOCONFIG
        os.environ["AWS_ACCESS_KEY_ID"] = AUTOCONFIG.aws_access_key()
        os.environ["AWS_SECRET_ACCESS_KEY"] = AUTOCONFIG.aws_access_secret()

    full_s3_path = os.path.join(AWS_S3_PATH, path)
    bucket_name, bucket_relative_path = split_s3_full_path(full_s3_path)
    command = "aws s3 sync s3://%s/%s %s" % (bucket_name, bucket_relative_path, local_path)
    print(command)
    stream = os.popen(command)
    output = stream.read()
    print(output)
    return local_path


def split_s3_full_path(s3_path):
    """
    Split "s3://foo/bar/baz" into "foo" and "bar/baz"
    """
    bucket_name_and_directories = s3_path.split('//')[1]
    bucket_name, *directories = bucket_name_and_directories.split('/')
    directory_path = '/'.join(directories)
    return bucket_name, directory_path


class CPU_Unpickler(pickle.Unpickler):
    """Utility for loading a pickled model on CPU machine saved from a GPU"""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def load_local_or_remote_file(filepath, file_type=None):
    local_path = local_path_from_s3_or_local_path(filepath)
    if file_type is None:
        extension = local_path.split('.')[-1]
        if extension == 'npy':
            file_type = NUMPY
        elif extension == 'pkl':
            file_type = PICKLE
        elif extension == 'joblib':
            file_type = JOBLIB
        else:
            raise ValueError("Could not infer file type.")
    if file_type == NUMPY:
        object = np.load(open(local_path, "rb"), allow_pickle=True)
    elif file_type == JOBLIB:
        object = joblib.load(local_path)
    else:
        #f = open(local_path, 'rb')
        #object = CPU_Unpickler(f).load()
        object = pickle.load(open(local_path, "rb"))
    print("loaded", local_path)
    return object


def get_absolute_path(path):
    if path[0] == "/":
        return path
    else:
        is_docker = os.path.isfile("/.dockerenv")
        if is_docker:
            local_path = "/tmp/%s" % (path)
        else:
            local_path = "%s/%s" % (LOCAL_LOG_DIR, path)
        return local_path


if __name__ == "__main__":
    p = sync_down("ashvin/vae/new-point2d/run0/id1/params.pkl")
    print("got", p)
