from gym.envs.mujoco import HalfCheetahEnv
import numpy as np

def test_mujoco_env_hc():
    env = HalfCheetahEnv()
    env.seed(0)

    obs = env.reset()
    print("reset observation")
    print(list(obs))
    expected_result = np.array([0.09307818744846408, 0.026538189116820654, -0.04199653123045835, -0.07950314858277296, 0.03461526961155151, -0.02148465198872647, 0.03396921345769463, -0.08803220624881061, -0.05727510457667884, 0.0033623362111972334, -0.003173373363900766, 0.04092179272969336, 0.05138553542843286, 0.01490225253385475, -0.04298222544789362, 0.1840846875514338, -0.004267549831952004])
    assert np.isclose(obs, expected_result).all()

    obs, _, _, _ = env.step(np.zeros(6,))
    print("after 1 step")
    print(list(obs))
    expected_result = np.array([0.07598225250646283, 0.019175277630375635, -0.026049271400655116, -0.02524626742505337, 0.0022975114812461717, -0.0022682205794546627, 0.012561418192984201, -0.03680577283317282, 0.011517471035147164, -0.5486275103205266, -0.18959104789073566, 0.5238752657158512, 1.2951048434195913, -0.7176542510873759, 0.4801683485682522, -0.5874709352139998, 1.4057385188694944])
    assert np.isclose(obs, expected_result).all()

    np.random.seed(0)
    env.reset()
    N = 1000
    M = 17
    observations = np.zeros((N, M))
    expected_observations = np.load("tests/regression/random/test_mujoco_env_obs.npy")
    for i in range(1000):
        obs, _, _, _ = env.step(np.random.random(6))
        assert np.isclose(obs, expected_observations[i, :]).all(), "observation %d diverged" % i
        observations[i, :] = obs
    print("results matched after %d steps" % N)
    # np.save("tests/regression/random/test_mujoco_env_obs.npy", observations)

if __name__ == "__main__":
    test_mujoco_env_hc()
