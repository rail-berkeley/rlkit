from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
import numpy as np
import torch

def test_mlp_run():
    torch.manual_seed(0)
    f = Mlp(hidden_sizes=[100, 100, 100], output_size=20, input_size=10)
    x = ptu.from_numpy(np.ones(10,))
    y = ptu.get_numpy(f(x))
    print(list(y))
    expected_result = np.array([-0.00051861757, -0.0002842698, 0.0009188504, 0.0002975871, 0.00061783846, -0.0004398331, -0.00053419394, -6.92709e-06, -0.00042290357, -6.476462e-05, 0.00031131462, -0.00036673213, -0.0004935546, -4.3982916e-05, -0.00041407614, -0.00028983108, 0.00072777556, -2.5328445e-05, -0.00015854833, -0.00013315874])
    assert np.isclose(expected_result, y).all()

if __name__ == "__main__":
    test_mlp_run()
