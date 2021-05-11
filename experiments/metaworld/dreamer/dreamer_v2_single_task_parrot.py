import argparse
import random
import subprocess

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
    num_envs = 10
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=5,
            num_eval_steps_per_epoch=1000,
            min_num_steps_before_training=1000,
            num_pretrain_steps=1,
            max_path_length=500,
            num_expl_steps_per_train_loop=501,
            num_trains_per_train_loop=1,
            num_train_loops_per_epoch=1,
            batch_size=50,
        )
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=100,
            num_eval_steps_per_epoch=500 * 5,
            min_num_steps_before_training=2500,
            num_pretrain_steps=100,
            max_path_length=500,
            num_expl_steps_per_train_loop=501 * num_envs,
            num_trains_per_train_loop=2000,
            num_train_loops_per_epoch=2,
            batch_size=50,
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="DreamerV2",
        version="normal",
        replay_buffer_size=int(5e3),
        algorithm_kwargs=algorithm_kwargs,
        pass_render_kwargs=False,
        env_suite="metaworld",
        use_raw_actions=True,
        env_kwargs=dict(
            control_mode="end_effector",
            use_combined_action_space=True,
            action_scale=1 / 100,
            max_path_length=500,
            reward_type="sparse",
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                use_image_obs=True,
                max_path_length=500,
                unflatten_images=False,
                use_real_nvp_wrappers=True,
                # model_path="/home/mdalal/research/rlkit/data/railrl-realNVP-slide-cabinet-pixels/railrl-realNVP-slide_cabinet-pixels_2021_05_07_21_20_56_id306830--s1975097/params.pkl",
            ),
            image_kwargs=dict(imwidth=64, imheight=64),
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
        ),
        num_expl_envs=num_envs,
        num_eval_envs=1,
        expl_amount=0.3,
    )

    search_space = {
        "env_name": [
            "basketball-v2",
            "assembly-v2",
            "soccer-v2",
            "sweep-into-v2",
            "disassemble-v2"
            "drawer-close-v2",
        ],
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
