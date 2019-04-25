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


def split_s3_full_path(s3_path):
    """
    Split "s3://foo/bar/baz" into "foo" and "bar/baz"
    """
    bucket_name_and_directories = s3_path.split('//')[1]
    bucket_name, *directories = bucket_name_and_directories.split('/')
    directory_path = '/'.join(directories)
    return bucket_name, directory_path


def load_local_or_remote_file(filepath, file_type=None):
    local_path = local_path_from_s3_or_local_path(filepath)
    if file_type is None:
        extension = local_path.split('.')[-1]
        if extension == 'npy':
            file_type = NUMPY
        else:
            file_type = PICKLE
    else:
        file_type = PICKLE
    if file_type == NUMPY:
        object = np.load(open(local_path, "rb"))
    elif file_type == JOBLIB:
        object = joblib.load(local_path)
    else:
        object = pickle.load(open(local_path, "rb"))
    print("loaded", local_path)
    return object


if __name__ == "__main__":
    p = sync_down("ashvin/vae/new-point2d/run0/id1/params.pkl")
    print("got", p)