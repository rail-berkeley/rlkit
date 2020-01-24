# Skew-Fit
Requires [multiworld](https://github.com/vitchyr/multiworld) to be installed:
```
pip install git+https://github.com/vitchyr/multiworld.git@28ee206f60a45690d484737466b558abdef191ea
```

Implementation of Skew-Fit. For more information:
 - [Videos](https://sites.google.com/view/skew-fit)
 - [arXiv](https://arxiv.org/abs/1903.03698)
 
Here are the results you should expect from each script.
These plots are generated with [viskit](https://github.com/vitchyr/viskit) 
with smoothing on.

Note that [RIG](RIG.md) is a special-case of Skew-Fit with `power=0`.


[examples/skewfit/sawyer_door.py](../examples/skewfit/sawyer_door.py). 1 Seed:
![Skew-Fit Sawyer Door results](images/skewfit_door.png)

[examples/skewfit/sawyer_pickup.py](../examples/skewfit/sawyer_pickup.py). 3 Seeds:
![Skew-Fit Sawyer Pickup results](images/skewfit_pickup.png)

[examples/skewfit/sawyer_pusher.py](../examples/skewfit/sawyer_pusher.py). 9 Seeds:
![Skew-Fit Sawyer Pusher results](images/skewfit_pusher.png)
