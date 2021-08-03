import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument("--logdir", type=str, default="clone_primitives")
    parser.add_argument("--datafile", type=str, default="data")
    parser.add_argument("--input_subselect", type=str, default="all")
    parser.add_argument("--num_actions", type=int, default=int(1e5))
    parser.add_argument("--num_epochs", type=int, default=int(1e1))
    parser.add_argument("--batch_size", type=int, default=int(256))
    parser.add_argument("--lr", type=float, default=float(1e-3))
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[64, 64])

    # parse arguments
    args = parser.parse_args()
    return args
