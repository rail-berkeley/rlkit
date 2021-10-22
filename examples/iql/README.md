# Implicit Q-Learning

This repository contains a PyTorch re-implementation of [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169) by [Ilya Kostrikov](https://kostrikov.xyz), [Ashvin Nair](https://ashvin.me/), and [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/).

For the official repository, please use: https://github.com/ikostrikov/implicit_q_learning

This code can be used for offline RL, or for offline RL followed by online finetuning. Negative epochs are offline RL, positive epochs are online (the agent is actively collecting data and adding it to the replay buffer).

If you use this code for your research, please consider citing the paper:
```
@article{kostrikov2021iql,
    title={Offline Reinforcement Learning with Implicit Q-Learning},
    author={Ilya Kostrikov and Ashvin Nair and Sergey Levine},
    year={2021},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Tests

To run quick versions of these experiments to test if the code matches exactly as the results below, you can run the tests in `tests/regression/iql`

## Mujoco results
![Mujoco results](https://i.ibb.co/6Pd8KT7/download-79.png)

## Antmaze results
![Ant-maze results](https://i.ibb.co/HrTMY2P/download-77.png)
