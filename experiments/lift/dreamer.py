import torch
import rlkit.util.hyperparameter as hyp
import argparse
import json
import libtmux
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--tmux", action="store_true", default=False)
    parser.add_argument("--tmux_session_name", type=str, default="")
    args = parser.parse_args()

    if args.tmux:
        server = libtmux.Server()
        session = server.find_where({"session_name": args.tmux_session_name})
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=2,
            num_eval_steps_per_epoch=30,
            num_trains_per_train_loop=10,
            num_expl_steps_per_train_loop=150,  # 200 samples since num_envs = 50 and max_path_length + 1 = 4
            min_num_steps_before_training=100,
            num_pretrain_steps=100,
            num_train_loops_per_epoch=1,
            max_path_length=3,
            batch_size=50,
        )
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=100,
            num_eval_steps_per_epoch=30,
            num_trains_per_train_loop=200,
            num_expl_steps_per_train_loop=150,  # 200 samples since num_envs = 50 and max_path_length + 1 = 4
            min_num_steps_before_training=5000,
            num_pretrain_steps=100,
            num_train_loops_per_epoch=5,
            max_path_length=3,
            batch_size=625,
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="Dreamer",
        version="normal",
        replay_buffer_size=int(1e5),
        algorithm_kwargs=algorithm_kwargs,
        env_class="lift",
        env_kwargs=dict(
            block_distance_to_lift=0,
            n_train_envs=50,
            n_eval_envs=10,
            fixed_schema=True,
            randomize_block_pose_on_reset=False,
            ee_lower=(0.45, 0.525, -0.05),
            ee_upper=(0.55, 0.625, 0.05),
        ),
        actor_kwargs=dict(
            split_dist=False,
        ),
        model_kwargs=dict(
            model_hidden_size=400,
            stochastic_state_size=60,
            deterministic_state_size=400,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            reward_scale=1.0,
            actor_lr=8e-5,
            vf_lr=8e-5,
            world_model_lr=6e-4,
            use_amp=False,
            opt_level="O1",
            gradient_clip=100.0,
            lam=0.95,
            imagination_horizon=algorithm_kwargs["max_path_length"] + 1,
            free_nats=3.0,
            kl_scale=1.0,
            optimizer_class="apex_adam",
            pcont_scale=10.0,
            use_pcont=True,
        ),
    )

    search_space = {
        # 'env_kwargs.ee_lower':[(.45, .525, -.05), (.4, .525, -.1), (.35, .525, -.15)],
        # 'env_kwargs.ee_upper':[(.55, .625, .05), (.6, .725, .1), (.65, .825, .15)],
        # 'env_kwargs.block_distance_to_lift':[0, .05, .1]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )

    num_gpus = torch.cuda.device_count()
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        gpu_id = exp_id % num_gpus  # only matters for docker runs
        json_var = json.dumps(variant)
        cmd = "python experiments/runner.py --variant '{}' --exp_prefix {} --mode {} --num_seeds {} --gpu_id {}".format(
            json_var,
            args.exp_prefix,
            args.mode,
            args.num_seeds,
            gpu_id,
        )
        if args.tmux:
            cmd = cmd + " --tmux_session_name " + args.tmux_session_name
            w = session.new_window(
                attach=False, window_name="exp_id:{} device:{}".format(exp_id, gpu_id)
            )
            pane = w.split_window()
            pane.send_keys("conda activate hrl-exp-env")
            pane.send_keys(cmd)
        else:
            os.system(cmd)
