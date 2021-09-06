# Regression Tests

The purpose of these tests is mainly to 1. catch bugs being introduced that cause past experiments/projects to fail 2. after code set up, know that the experiment you are running is exactly the one eg. reported in a paper.

Tests may be run by individually running an individual file with `python`, or `pip install nose2` and run:
```
nose2 -v -B -s tests/regression/<folder>
```

The tests are divided into different project folders, and some require specific software setup. They are described below:

### random

Tests whether stochasticity is fully controlled and match the source tests. Unfortunately if `tests/regression/random/test_mujoco_env.py` fails then any tests that collect online data with MuJoCo environments are likely to also fail.

### basic

Tests whether the basic RL algorithm reference scripts run and the results exactly match with smaller batch sizes and epochs.

### sac

Tests SAC - this is redundant with `basic` but a different style of test.

### awac

Tests AWAC, which runs offline RL followed by online finetuning, in different domains. Any tests that are "offline" do not collect additional data in the environment and just test the algorithm. In particular `gcrl/test_awac_gcrl_offline.py` is a quick test to run the data is included in the repo.

### val

Tests VAL, split into two parts: one tests training the VQVAE and one tests running RL with a pretrained VQVAE.
