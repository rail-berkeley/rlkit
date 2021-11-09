import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument("--logdir", type=str, default="clone_primitives")
    parser.add_argument("--datafile", type=str, default="data")
    parser.add_argument("--input_subselect", type=str, default="all")
    parser.add_argument("--num_trajs", type=int, default=int(500))
    parser.add_argument("--num_epochs", type=int, default=int(1e1))
    parser.add_argument("--batch_size", type=int, default=int(50))
    parser.add_argument("--batch_len", type=int, default=int(50))
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--max_path_length", type=int, default=500)
    parser.add_argument("--num_envs", type=int, default=5)
    parser.add_argument("--train_test_split", type=float, default=0.8)
    parser.add_argument("--control_mode", type=str, default="end_effector")
    parser.add_argument("--use_prior_instead_of_posterior", type=bool, default=False)
    parser.add_argument("--num_low_level_actions_per_primitive", type=int, default=100)

    # parse arguments
    args = parser.parse_args()
    return args
