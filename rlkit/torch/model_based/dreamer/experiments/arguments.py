import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Experiment Launcher Arguments")

    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--debug", action="store_true", default=False)

    # parse arguments
    args = parser.parse_args()
    if args.debug:
        args.exp_prefix = "test" + args.exp_prefix

    return args
