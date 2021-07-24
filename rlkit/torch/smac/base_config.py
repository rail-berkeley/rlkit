DEFAULT_CONFIG = {
    "qf_kwargs": {
        "hidden_sizes": [300, 300, 300],
    },
    "policy_kwargs": {
        "hidden_sizes": [300, 300, 300],
    },
    "logger_config": {
        "snapshot_mode": "gap_and_last",
        "snapshot_gap": 25,
    },
    "context_decoder_kwargs": {
        "hidden_sizes": [64, 64],
    },
    "save_video": False,
    "save_video_period": 25,

    "pretrain_rl": True,

    "trainer_kwargs": {
        "beta": 100.0,
        "alpha": 0.0,
        "rl_weight": 1.0,
        "use_awr_update": True,
        "use_reparam_update": False,
        "use_automatic_entropy_tuning": False,
        "awr_weight": 1.0,
        "bc_weight": 0.0,
        "compute_bc": False,
        "awr_use_mle_for_vf": False,
        "awr_sample_actions": False,
        "awr_min_q": True,
        "reparam_weight": 0.0,
        "backprop_q_loss_into_encoder": False,
        "train_context_decoder": True,

        "soft_target_tau": 0.005,  # for SAC target network update
        "target_update_period": 1,
        "policy_lr": 3E-4,
        "qf_lr": 3E-4,
        "context_lr": 3e-4,
        "kl_lambda": .1,  # weight on KL divergence term in encoder loss
        "use_information_bottleneck": True, # False makes latent context deterministic
        "use_next_obs_in_context": False, # use next obs if it is useful in distinguishing tasks
        "sparse_rewards": False, # whether to sparsify rewards as determined in env
        "recurrent": False, # recurrent or permutation-invariant encoder
        "discount": 0.99, # RL discount factor
        "reward_scale": 5.0, # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
    },
    "tags": {},
    "latent_dim": 5,
    "algo_kwargs": {
        "use_rl_buffer_for_enc_buffer": True,
        "freeze_encoder_buffer_in_unsupervised_phase": False,
        "clear_encoder_buffer_before_every_update": False,
        "num_iterations_with_reward_supervision": 0,
        "exploration_resample_latent_period": 1,
        "meta_batch": 4,
        "embedding_batch_size": 256,
        "num_initial_steps": 2000,
        "num_steps_prior": 400,
        "num_steps_posterior": 0,
        "num_extra_rl_steps_posterior": 600,
        "num_train_steps_per_itr": 4000,
        "num_evals": 4,
        "num_steps_per_eval": 600,
        "num_exp_traj_eval": 2,

        "num_iterations": 501, # number of data sampling / training iterates
        "num_tasks_sample": 5, # number of randomly sampled tasks to collect data for each iteration
        "batch_size": 256, # number of transitions in the RL batch
        "embedding_mini_batch_size": 64, # number of context transitions to backprop through (should equal the arg above except in the recurrent encoder case)
        "max_path_length": 200, # max path length for this environment
        "update_post_train": 1, # how often to resample the context when collecting data during training (in trajectories)
        "dump_eval_paths": False, # whether to save evaluation trajectories
        "save_extra_manual_epoch_list": [0, 50, 100, 200, 300, 400, 500],
        "save_extra_manual_beginning_epoch_list": [0],
        "save_replay_buffer": False,
        "save_algorithm": True,
    },
    "online_trainer_kwargs": {
        "awr_weight": 1.0,
        "reparam_weight": 1.0,
        "use_reparam_update": True,
        "use_awr_update": True,
    },
    "skip_initial_data_collection_if_pretrained": True,
    "pretrain_offline_algo_kwargs": {
        "batch_size": 128,
        "logging_period": 1000,
        "meta_batch_size": 4,
        "num_batches": 50000,
        "task_embedding_batch_size": 64,
    },
    "n_train_tasks": 100,
    "n_eval_tasks": 20,
    "env_params": {},
}

DEFAULT_PEARL_CONFIG = {
    "qf_kwargs": {
        "hidden_sizes": [300, 300, 300],
    },
    "vf_kwargs": {
        "hidden_sizes": [300, 300, 300],
    },
    "policy_kwargs": {
        "hidden_sizes": [300, 300, 300],
    },
    "trainer_kwargs": {
        "soft_target_tau": 0.005,  # for SAC target network update
        "target_update_period": 1,
                              "policy_lr": 3E-4,
        "qf_lr": 3E-4,
        "context_lr": 3e-4,
        "kl_lambda": .1,  # weight on KL divergence term in encoder loss
        "use_information_bottleneck": True, # False makes latent context deterministic
        "use_next_obs_in_context": False, # use next obs if it is useful in distinguishing tasks
        "sparse_rewards": False, # whether to sparsify rewards as determined in env
        "recurrent": False, # recurrent or permutation-invariant encoder
        "discount": 0.99, # RL discount factor
        "reward_scale": 5.0, # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
        "backprop_q_loss_into_encoder": True,
    },
    "algo_kwargs": {
        "num_iterations": 501, # number of data sampling / training iterates
        "num_tasks_sample": 5, # number of randomly sampled tasks to collect data for each iteration
        "batch_size": 256, # number of transitions in the RL batch
        "embedding_mini_batch_size": 64, # number of context transitions to backprop through (should equal the arg above except in the recurrent encoder case)
        "max_path_length": 200, # max path length for this environment
        "update_post_train": 1, # how often to resample the context when collecting data during training (in trajectories)
        "dump_eval_paths": False, # whether to save evaluation trajectories
        "num_iterations_with_reward_supervision": None,
        "save_extra_manual_epoch_list": [0, 50, 100, 200, 300, 400, 500],
        "save_extra_manual_beginning_epoch_list": [0],
        "save_replay_buffer": False,
        "save_algorithm": True,
        "exploration_resample_latent_period": 1,

        "freeze_encoder_buffer_in_unsupervised_phase": False,
        "clear_encoder_buffer_before_every_update": True,
        "meta_batch": 4,
        "embedding_batch_size": 256,
        "num_initial_steps": 2000,
        "num_steps_prior": 400,
        "num_steps_posterior": 0,
        "num_extra_rl_steps_posterior": 600,
        "num_train_steps_per_itr": 4000,
        "num_evals": 4,
        "num_steps_per_eval": 600,
        "num_exp_traj_eval": 2,
    },
    "latent_dim": 5,
    "logger_config": {
        "snapshot_mode": "gap_and_last",
        "snapshot_gap": 25,
    },
    "context_decoder_kwargs": {
        "hidden_sizes": [64, 64],
    },
    "env_params": {},
    "n_train_tasks": 100,
    "n_eval_tasks": 20,
}
