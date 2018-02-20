import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.modules import HuberLoss
from rlkit.torch.tdm.envs.reacher_7dof_env import Reacher7DofFullGoal
from rlkit.torch.tdm.her_replay_buffer import HerReplayBuffer
from rlkit.torch.tdm.networks import TdmNormalizer, TdmPolicy, TdmQf
from rlkit.torch.tdm.tdm import TemporalDifferenceModel


def experiment(variant):
    env = NormalizedBoxEnv(Reacher7DofFullGoal())
    max_tau = variant['tdm_kwargs']['max_tau']
    # Normalizer isn't used unless you set num_pretrain_paths > 0
    tdm_normalizer = TdmNormalizer(
        env,
        vectorized=True,
        max_tau=max_tau,
    )
    qf = TdmQf(
        env=env,
        vectorized=True,
        norm_order=1,
        tdm_normalizer=tdm_normalizer,
        hidden_sizes=[300, 300],
    )
    policy = TdmPolicy(
        env=env,
        tdm_normalizer=tdm_normalizer,
        hidden_sizes=[300, 300],
    )
    es = OUStrategy(
        action_space=env.action_space,
        theta=0.1,
        max_sigma=0.1,
        min_sigma=0.1,
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer = HerReplayBuffer(
        env=env,
        max_size=int(2E4),
    )
    algorithm = TemporalDifferenceModel(
        env,
        qf=qf,
        replay_buffer=replay_buffer,
        policy=policy,
        exploration_policy=exploration_policy,
        qf_criterion=HuberLoss(),
        tdm_normalizer=tdm_normalizer,
        **variant['tdm_kwargs']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        tdm_kwargs=dict(
            # TDM parameters
            max_tau=10,
            num_pretrain_paths=0,

            # General parameters
            num_epochs=100,
            num_steps_per_epoch=100,
            num_steps_per_eval=1000,
            max_path_length=100,
            num_updates_per_env_step=25,
            batch_size=128,
            discount=1,
            reward_scale=100,

            # DDPG soft-target tau (not TDM tau)
            tau=0.001,
        ),
        algorithm="TDM",
    )
    setup_logger('name-of-tdm-reacher-experiment', variant=variant)
    experiment(variant)
