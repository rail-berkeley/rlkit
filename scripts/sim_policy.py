from rlkit.envs.remote import RemoteRolloutEnv
from rlkit.samplers.util import rollout
from rlkit.torch.core import PyTorchModule
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import joblib
import uuid
from rlkit.core import logger


filename = str(uuid.uuid4())


def simulate_policy(args):
    data = joblib.load(args.file)
    if 'policy' in data:
        policy = data['policy']
    elif 'exploration_policy' in data:
        policy = data['exploration_policy']
    elif 'naf_policy' in data:
        policy = data['naf_policy']
    elif 'optimizable_qfunction' in data:
        qf = data['optimizable_qfunction']
        policy = qf.implicit_policy
    else:
        raise Exception("No policy found in loaded dict. Keys: {}".format(
            data.keys()
        ))

    env = data['env']
    if isinstance(env, RemoteRolloutEnv):
        env = env._wrapped_env
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    if args.pause:
        import ipdb; ipdb.set_trace()
    if isinstance(policy, PyTorchModule):
        policy.train(False)
    while True:
        try:
            path = rollout(
                env,
                policy,
                max_path_length=args.H,
                animated=True,
            )
            if hasattr(env, "log_diagnostics"):
                env.log_diagnostics([path])
            policy.log_diagnostics([path])
            logger.dump_tabular()
        # Hack for now. Not sure why rollout assumes that close is an
        # keyword argument
        except TypeError as e:
            if (str(e) != "render() got an unexpected keyword "
                          "argument 'close'"):
                raise e

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)

    # TODO(vitchyr): maybe add this check back in with a try-except statement
    # import tensorflow as tf
    # with tf.Session() as sess:
    #     simulate_policy(args)
