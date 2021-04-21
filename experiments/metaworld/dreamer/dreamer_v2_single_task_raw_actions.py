import argparse
import random

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.model_based.dreamer.experiments.kitchen_dreamer import experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    num_envs = 11
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=5,
            num_eval_steps_per_epoch=1000,
            min_num_steps_before_training=1000,
            num_pretrain_steps=1,
            max_path_length=150,
            num_expl_steps_per_train_loop=151,
            num_trains_per_train_loop=1,
            num_train_loops_per_epoch=1,
            batch_size=50,
        )
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=250,
            num_eval_steps_per_epoch=150 * 5,
            min_num_steps_before_training=2500,
            num_pretrain_steps=100,
            max_path_length=150,
            num_expl_steps_per_train_loop=150 * num_envs,
            num_trains_per_train_loop=333,
            num_train_loops_per_epoch=6,
            batch_size=50,
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="DreamerV2",
        version="normal",
        replay_buffer_size=int(5e3),
        algorithm_kwargs=algorithm_kwargs,
        env_suite="metaworld",
        use_raw_actions=True,
        env_kwargs=dict(
            control_mode="end_effector",
            use_combined_action_space=False,
            action_scale=1 / 100,
        ),
        actor_kwargs=dict(
            init_std=0.0,
            num_layers=4,
            min_std=0.1,
            dist="trunc_normal",
        ),
        vf_kwargs=dict(
            num_layers=3,
        ),
        model_kwargs=dict(
            model_hidden_size=400,
            stochastic_state_size=50,
            deterministic_state_size=200,
            embedding_size=1024,
            rssm_hidden_size=200,
            reward_num_layers=2,
            pred_discount_num_layers=3,
            gru_layer_norm=True,
            std_act="sigmoid2",
        ),
        trainer_kwargs=dict(
            use_amp=True,
            opt_level="O1",
            optimizer_class="apex_adam",
            adam_eps=1e-5,
            discount=0.99,
            lam=0.95,
            forward_kl=False,
            free_nats=1.0,
            kl_loss_scale=0.0,
            transition_loss_scale=0.8,
            actor_lr=8e-5,
            vf_lr=8e-5,
            world_model_lr=3e-4,
            reward_loss_scale=2.0,
            imagination_horizon=15,
            use_pred_discount=False,
            policy_gradient_loss_scale=0.0,
            actor_entropy_loss_schedule="1e-4",
            target_update_period=100,
            reward_scale=1 / 100,
        ),
        num_expl_envs=num_envs,
        num_eval_envs=1,
        expl_amount=0.3,
    )

    search_space = {
        "env_class": [
            # "assembly-v2",  # hard envs
            # "basketball-v2",  # hard envs
            # "bin-picking-v2",  # hard envs
            # # "box-close-v2",
            # # "button-press-topdown-v2",
            # # "button-press-topdown-wall-v2",
            # # "button-press-v2",
            # # "button-press-wall-v2",
            # # "coffee-button-v2",
            # "coffee-pull-v2",  # hard envs
            # # "coffee-push-v2",
            # "dial-turn-v2",  # hard envs
            # "disassemble-v2",  # hard envs
            # "door-close-v2",  # hard envs
            # "door-lock-v2",
            # "door-open-v2",
            # "door-unlock-v2",
            # "hand-insert-v2", #no goal
            # "drawer-close-v2", #no goal
            # "drawer-open-v2", #no goal
            # "faucet-open-v2", #no goal
            # "faucet-close-v2", #no goal
            # "faucet-close-v2",
            # "hammer-v2",
            # "handle-press-side-v2",
            # "handle-press-v2",
            # "handle-pull-side-v2",
            # "handle-pull-v2",
            # "lever-pull-v2",
            # "peg-insert-side-v2",
            # "pick-place-wall-v2",
            # "pick-out-of-hole-v2",
            # "reach-v2",
            # "push-back-v2",
            # "push-v2",
            # "pick-place-v2",
            # "plate-slide-v2",
            # "plate-slide-side-v2",
            # "plate-slide-back-v2",
            # "plate-slide-back-side-v2",
            # "peg-unplug-side-v2",
            # "soccer-v2",
            # "stick-push-v2",
            # "stick-pull-v2",
            # "push-wall-v2",
            # "reach-wall-v2",
            # "shelf-place-v2",
            # "sweep-into-v2",
            # "sweep-v2",
            # "window-open-v2", #no goal
            # "window-close-v2", #no goal
            #
            # v1 envs:
            "reach-v1",
            "push-v1",
            "pick-place-v1",
            "door-open-v1",
            "drawer-open-v1",
            "drawer-close-v1",
            "button-press-topdown-v1",
            "peg-insert-side-v1",
            "window-open-v1",
            "window-close-v1",
            "door-close-v1",
            "reach-wall-v1",
            "pick-place-wall-v1",
            "push-wall-v1",
            "button-press-v1",
            "button-press-topdown-wall-v1",
            "button-press-wall-v1",
            "peg-unplug-side-v1",
            "disassemble-v1",
            # subset
            # "hammer-v1",
            # "plate-slide-v1",
            # "plate-slide-side-v1",
            # "plate-slide-back-v1",
            # "plate-slide-back-side-v1",
            # "handle-press-v1",
            # "handle-pull-v1",
            # "handle-press-side-v1",
            # "handle-pull-side-v1",
            # "stick-push-v1",
            # "stick-pull-v1",
            # "basketball-v1",
            # "soccer-v1",
            # "faucet-open-v1",
            # "faucet-close-v1",
            # "coffee-push-v1",
            # "coffee-pull-v1",
            # "coffee-button-v1",
            # "sweep-v1",
            # "sweep-into-v1",
            # "pick-out-of-hole-v1",
            # "assembly-v1",
            # "shelf-place-v1",
            # "push-back-v1",
            # "lever-pull-v1",
            # "dial-turn-v1",
            # "bin-picking-v1",
            # "box-close-v1",
            # # "hand-insert-v1",
            # "door-lock-v1",
            # "door-unlock-v1",
        ],
        # "trainer_kwargs.reward_scale": [1 / 100, 1 / 1000],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    num_exps_launched = 0
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(args.num_seeds):
            seed = random.randint(0, 100000)
            variant["seed"] = seed
            variant["exp_id"] = exp_id
            run_experiment(
                experiment,
                exp_prefix=args.exp_prefix,
                mode=args.mode,
                variant=variant,
                use_gpu=True,
                snapshot_mode="none",
                python_cmd=subprocess.check_output("which python", shell=True).decode(
                    "utf-8"
                )[:-1],
                seed=seed,
                exp_id=exp_id,
            )
            num_exps_launched += 1
    print("Num exps launched: ", num_exps_launched)
