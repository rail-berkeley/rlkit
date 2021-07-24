Requirements that differ from base requirements:
 - python 3.6.5
 - joblib==0.9.4
 - numpy==1.18.5
 
Running these examples requires first generating the data, updating the main launch script to point to that generated data, and then launching the SMAC experiments.

This can be done by first running
```bash
python examples/smac/generate_{ant|cheetah}_data.py
```
which runs [PEARL](https://github.com/katerakelly/oyster) to generate multi-task data.
This script will generate a directory and file of the form
```
LOCAL_LOG_DIR/<experiment_prefix>/<foldername>/extra_snapshot_itrXYZ.cpkl
```

You can then update the `examples/smac/{ant|cheetah}.py` file, where it says `TODO: update to point to correct file` to point to this file.
Finally, run the SMAC script
```bash
python examples/smac/{ant|cheetah}.py
```
