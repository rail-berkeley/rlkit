# default PEARL experiment settings
# all experiments should modify these settings only as needed
def make_variant(max_path_length):
    variant = dict(
        env_name='point-robot',
        n_train_tasks=5,
        n_eval_tasks=5,
        latent_size=5, # dimension of the latent context vector
        net_size=300, # number of units per FC layer in each network
        path_to_weights=None, # path to pre-trained weights to load into networks
        env_params=dict(
            n_tasks=10, # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
            randomize_tasks=True, # shuffle the tasks after creating them
        ),
        algo_params=dict(
            meta_batch=16, # number of tasks to average the gradient across
            num_iterations=10000, # number of data sampling / training iterates
            num_initial_steps=10 * max_path_length, # number of transitions collected per task before training
            num_tasks_sample=5, # number of randomly sampled tasks to collect data for each iteration
            num_steps_per_task=10 * max_path_length, # number of transitions to collect per task
            num_train_steps_per_itr=1000, # number of meta-gradient steps taken per iteration
            num_evals=5, # number of independent evals
            num_steps_per_eval=3 * max_path_length,  # nuumber of transitions to eval on
            batch_size=256, # number of transitions in the RL batch
            embedding_batch_size=64, # number of transitions in the context batch
            embedding_mini_batch_size=64, # number of context transitions to backprop through (should equal the arg above except in the recurrent encoder case)
            max_path_length=max_path_length, # max path length for this environment
            discount=0.99, # RL discount factor
            soft_target_tau=0.005, # for SAC target network update
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            context_lr=3e-4,
            reward_scale=100., # scale rewards before constructing Bellman update
            sparse_rewards=False, # whether to sparsify rewards as determined in env
            reparameterize=True, # should always be True
            kl_lambda=.1, # weight on KL divergence term in encoder loss
            use_information_bottleneck=True, # False makes latent context deterministic
            train_embedding_source='online_exploration_trajectories',
            # embedding_source should be chosen from
            # {'initial_pool', 'online_exploration_trajectories', 'online_on_policy_trajectories'}
            eval_embedding_source='online_exploration_trajectories',
            recurrent=False, # recurrent or permutation-invariant encoder
            dump_eval_paths=False, # whether to save evaluation trajectories
        ),
        util_params=dict(
            base_log_dir='output',
            use_gpu=True,
            gpu_id=0,
            debug=False, # debugging triggers printing and writes logs to debug directory
            docker=False, # TODO docker is not yet supported
        )
    )
    return variant



