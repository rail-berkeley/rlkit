"""Wraps launcher_util to make launching experiments one step easier - Ashvin
- Names experiments based on the running filename
- Adds some modes like --1 to run only one variant of a set for testing
- Control the GPU used and other experiment attributes through command line args
"""

from rlkit.launchers import launcher_util as lu
import argparse # TODO: migrate to argparse if necessary
import sys
from multiprocessing import Process, Pool
import pdb
import random

def run_variants(experiment, vs, process_args_fn=None, run_id=0, ):
    # preprocess
    variants = []
    for i, v in enumerate(vs):
        v["exp_id"] = i
        v["run_id"] = i
        process_run_args(v)
        process_logger_args(v, run_id=run_id)
        process_launcher_args(v)
        if process_args_fn:
            process_args_fn(v)
        variants.append(v)

    if "--variants" in sys.argv: # takes either 3-7 or 3,6,7,8,10 as next arg
        i = sys.argv.index("--variants")
        val = sys.argv[i+1]
        ids = []
        if "," in val:
            ids = map(int, val.split(','))
        elif "-" in val:
            start, end = map(int, val.split(','))
            ids = range(start, end)
        else:
            ids = [int(val), ]
        new_variants = []
        for v in variants:
            if v["exp_id"] in ids:
                new_variants.append(v)
        variants = new_variants
    if "--1" in sys.argv: # only run the first experiment for testing
        variants = variants[:1]

    print("Running", len(variants), "variants")

    # run
    parallel = variants[0].get("parallel", False)
    if parallel:
        parallel_run(experiment, variants, parallel)
    else:
        for variant in variants:
            run_variant(experiment, variant)

    print("Running", len(variants), "variants")

def run_variant(experiment, variant):
    launcher_config = variant.pop("launcher_config")
    lu.run_experiment(
        experiment,
        variant=variant,
        **launcher_config,
    )

def parallel_run(experiment, variants, n_p):
    i = 0
    # import pdb; pdb.set_trace()
    while i < len(variants):
        prs = []
        for p in range(n_p):
            if i < len(variants):
                v = variants[i]
                v["gpu_id"] = v["gpus"][p]
                pr = Process(target=run_variant, args=(experiment, v))
                prs.append(pr)
                pr.start()
            i += 1
        for pr in prs:
            pr.join()

def process_run_args(variant):
    if "--sync" in sys.argv:
        variant["sync"] = True
    if "--nosync" in sys.argv:
        variant["sync"] = False

    if "--render" in sys.argv:
        variant["render"] = True
        if "algo_kwargs" in variant:
            if "base_kwargs" in variant["algo_kwargs"]:
                variant["algo_kwargs"]["base_kwargs"]["render"] = True
    if "--norender" in sys.argv:
        variant["render"] = False
    if "--debug" in sys.argv:
        variant["debug"] = True

    if "--seed" in sys.argv:
        i = sys.argv.index("--seed")
        variant["seed"] = sys.argv[i+1]

    if "--parallel" in sys.argv:
        i = sys.argv.index("--parallel")
        parallel = int(sys.argv[i+1])
        variant["parallel"] = parallel
        if "--gpus" in sys.argv:
            i = sys.argv.index("--gpus")
            variant["gpus"] = list(map(int, sys.argv[i+1].split(",")))
            variant["use_gpu"] = True
        else:
            variant["gpus"] = list(range(parallel))


def process_logger_args(variant, run_id=None):
    logger_config = variant.setdefault("logger_config", dict())

    logger_config["snapshot_mode"] = logger_config.get("snapshot_mode", "gap")
    logger_config["snapshot_gap"] = logger_config.get("snapshot_gap", 100)
    if "--snapshot" in sys.argv:
        logger_config["snapshot_mode"] = 'gap_and_last'
        logger_config["snapshot_gap"] = 20
    elif "--nosnapshot" in sys.argv:
        logger_config["snapshot_mode"] = 'none'
        logger_config["snapshot_gap"] = 1

    if "--run" in sys.argv:
        i = sys.argv.index("--run")
        logger_config["run_id"] = int(sys.argv[i+1])
        variant["run_id"] = int(sys.argv[i+1])
    else:
        logger_config["run_id"] = run_id


def process_launcher_args(variant):
    launcher_config = variant.setdefault("launcher_config", dict())

    launcher_config.setdefault("gpu_id", 0)
    launcher_config.setdefault("prepend_date_to_exp_name", False)
    launcher_config.setdefault("region", "us-west-2")
    launcher_config.setdefault("time_in_mins", None)
    launcher_config.setdefault("ssh_host", None)
    launcher_config.setdefault("slurm_config_name", None)
    launcher_config.setdefault("unpack_variant", False)
    launcher_config.setdefault("s3_log_prefix", "")

    if "--ec2" in sys.argv:
        launcher_config["mode"] = "ec2"
    if "--local" in sys.argv:
        launcher_config["mode"] = "here_no_doodad"
    if "--localdocker" in sys.argv:
        launcher_config["mode"] = "local_docker"
    if "--sss" in sys.argv:
        launcher_config["mode"] = "sss"
    if "--singularity" in sys.argv:
        launcher_config["mode"] = "local_singularity"
    if "--slurm" in sys.argv:
        launcher_config["mode"] = "slurm"
    if "--ss" in sys.argv:
        launcher_config["mode"] = "slurm_singularity"
    if "--sss" in sys.argv:
        launcher_config["mode"] = "sss"
    if "--htp" in sys.argv:
        launcher_config["mode"] = "htp"
    if "--ssh" in sys.argv:
        launcher_config["mode"] = "ssh"
        i = sys.argv.index("--ssh")
        launcher_config["ssh_host"] = sys.argv[i+1]

    if "--slurmconfig" in sys.argv:
        i = sys.argv.index("--slurmconfig")
        launcher_config["slurm_config_name"] = sys.argv[i+1]
    if "--lowprio" in sys.argv:
        launcher_config["slurm_config_name"] = "gpu_lowprio"

    if "--verbose" in sys.argv:
        launcher_config["verbose"] = True

    if "--gpu_id" in sys.argv:
        i = sys.argv.index("--gpu_id")
        launcher_config["gpu_id"] = int(sys.argv[i+1])
        launcher_config["use_gpu"] = True
    if "--gpu" in sys.argv:
        launcher_config["use_gpu"] = True
    if "use_gpu" in launcher_config and launcher_config["use_gpu"]:
        if "instance_type" not in launcher_config:
            launcher_config["instance_type"] = "g3.4xlarge"
    if "--time" in sys.argv:
        i = sys.argv.index("--time")
        launcher_config["time_in_mins"] = int(sys.argv[i+1])

    if "instance_type" not in launcher_config:
        launcher_config["instance_type"] = "c4.xlarge"
    if "use_gpu" not in launcher_config:
        launcher_config["use_gpu"] = None

    if "base_log_dir" not in launcher_config:
        launcher_config["base_log_dir"] = None
    if "--mac" in sys.argv:
        launcher_config["base_log_dir"] = "/Users/ashvin/data/s3doodad/"

    if "exp_name" not in launcher_config:
        launcher_config["exp_name"] = sys.argv[0][:-3]

