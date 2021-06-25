from collections import OrderedDict, namedtuple

from rlkit.torch.model_based.dreamer.dreamer_v2 import DreamerV2Trainer

DreamerLosses = namedtuple(
    "DreamerLosses",
    "actor_loss vf_loss world_model_loss",
)


class DreamerTrainer(DreamerV2Trainer):
    def __init__(
        self,
        env,
        actor,
        vf,
        world_model,
        imagination_horizon,
        image_shape,
        target_vf=None,
        discount=0.99,
        reward_scale=1.0,
        actor_lr=8e-5,
        vf_lr=8e-5,
        world_model_lr=6e-4,
        gradient_clip=100.0,
        lam=0.95,
        free_nats=3.0,
        kl_loss_scale=1.0,
        pred_discount_loss_scale=10.0,
        adam_eps=1e-7,
        weight_decay=0.0,
        use_pred_discount=True,
        debug=False,
    ):
        super(DreamerTrainer, self).__init__(
            env,
            actor,
            vf,
            world_model,
            imagination_horizon,
            image_shape,
            target_vf=target_vf,
            discount=discount,
            reward_scale=reward_scale,
            actor_lr=actor_lr,
            vf_lr=vf_lr,
            world_model_lr=world_model_lr,
            world_model_gradient_clip=gradient_clip,
            actor_gradient_clip=gradient_clip,
            value_gradient_clip=gradient_clip,
            lam=lam,
            free_nats=free_nats,
            kl_loss_scale=kl_loss_scale,
            pred_discount_loss_scale=pred_discount_loss_scale,
            adam_eps=adam_eps,
            weight_decay=weight_decay,
            use_pred_discount=use_pred_discount,
            debug=debug,
            image_loss_scale=1.0,
            reward_loss_scale=1.0,
            transition_loss_scale=0.0,
            entropy_loss_scale=0.0,
            forward_kl=True,
            policy_gradient_loss_scale=0.0,
            actor_entropy_loss_schedule="0.0",
            soft_target_tau=1,
            target_update_period=1,
        )
