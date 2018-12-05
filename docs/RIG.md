# Reinforcement Learning with Imagined Goals
Implementation of
RIG. To find out more, see any of the following links:
* arXiv: https://arxiv.org/abs/1807.04742
* Website: https://sites.google.com/site/visualrlwithimaginedgoals/
* Blog Post: https://bair.berkeley.edu/blog/2018/09/06/rig/

For example scripts, see [this example script](examples/rig/pusher/rig.py)
for the pushing task, or
[this script on a simplified pointmass
environment](examples/rig/pointmass/rig.py).
The pushing task will take a long time initially as it generates an image
dataaset to pre-train a VAE. However, we also found that we can train the VAE
online without hurting performance.

## Goal-based environments and `ObsDictRelabelingBuffer`
[See here.](goal_based_envs.md)
