# Reinforcement Learning with Imagined Goals
Implementation of
RIG.
* arXiv: https://arxiv.org/abs/1807.04742
* Website: https://sites.google.com/site/visualrlwithimaginedgoals/
* Blog Post: https://bair.berkeley.edu/blog/2018/09/06/rig/

## Expected Results
If you run the [pointmass examples](examples/rig/pointmass/rig.py), 
then you should get results like this:
 ![Pointmass RIG results](images/pointmass-rig.png)
 
If you run the [pusher examples](examples/rig/pusher/rig.py), 
then you should get results like this:
 ![Pusher RIG results](images/pusher-rig.png)
 
Note that these examples use HER combined with TD3, and not DDPG.
TD3 is a new method that came out after the HER paper, and it seems to work 
better than DDPG.

## Goal-based environments and `ObsDictRelabelingBuffer`
[See here.](goal_based_envs.md)