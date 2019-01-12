import argparse
import json

import joblib
from pathlib import Path

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import get_generic_path_information
from rlkit.torch.tdm.sampling import multitask_rollout
from rlkit.core import logger
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--nrolls', type=int, default=1,
                        help='Number of rollout per eval')
    parser.add_argument('--mtau', type=float, help='Max tau value')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--hide', action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
    if args.mtau is None:
        # Load max tau from variant.json file
        variant_path = Path(args.file).parents[0] / 'variant.json'
        variant = json.load(variant_path.open())
        try:
            max_tau = variant['tdm_kwargs']['max_tau']
            print("Max tau read from variant: {}".format(max_tau))
        except KeyError:
            print("Defaulting max tau to 0.")
            max_tau = 0
    else:
        max_tau = args.mtau

    env = data['env']
    policy = data['policy']
    policy.train(False)

    if args.gpu:
        ptu.set_gpu_mode(True)
        policy.cuda()

    while True:
        paths = []
        for _ in range(args.nrolls):
            goal = env.sample_goal_for_rollout()
            path = multitask_rollout(
                env,
                policy,
                init_tau=max_tau,
                goal=goal,
                max_path_length=args.H,
                animated=not args.hide,
                cycle_tau=True,
                decrement_tau=True,
            )
            paths.append(path)
        env.log_diagnostics(paths)
        for key, value in get_generic_path_information(paths).items():
            logger.record_tabular(key, value)
        logger.dump_tabular()
