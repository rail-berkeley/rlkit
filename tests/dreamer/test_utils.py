import torch

from rlkit.torch.model_based.dreamer.utils import lambda_return


def test_lambda_return_zero():
    horizon = 500
    batch_size = 1
    reward = torch.zeros(horizon - 1, batch_size, 1)
    value = torch.zeros(horizon - 1, batch_size, 1)
    discount = torch.ones(horizon - 1, batch_size, 1)
    bootstrap = torch.zeros(batch_size, 1)
    lambda_ = 0
    returns = lambda_return(reward, value, discount, bootstrap, lambda_)
    assert returns.shape == (horizon - 1, batch_size, 1)
    assert returns.sum().item() == 0


def test_lambda_return_one():
    horizon = 2
    batch_size = 1
    reward = torch.zeros(horizon - 1, batch_size, 1)
    value = torch.zeros(horizon - 1, batch_size, 1)
    discount = torch.ones(horizon - 1, batch_size, 1)
    bootstrap = torch.ones(batch_size, 1)
    lambda_ = 0
    returns = lambda_return(reward, value, discount, bootstrap, lambda_)
    assert returns.shape == (horizon - 1, batch_size, 1)
    assert returns.item() == 1


def test_lambda_return_one_step_return():
    horizon = 2
    batch_size = 1
    reward = torch.rand(horizon - 1, batch_size, 1)
    value = torch.zeros(horizon - 1, batch_size, 1)
    discount = torch.ones(horizon - 1, batch_size, 1)
    bootstrap = torch.zeros(batch_size, 1)
    lambda_ = 1
    returns = lambda_return(reward, value, discount, bootstrap, lambda_)
    assert returns.shape == (horizon - 1, batch_size, 1)
    assert returns.sum().item() == reward.sum().item()


def test_lambda_return_one_step_return():
    horizon = 3
    batch_size = 1
    reward = torch.rand(horizon - 1, batch_size, 1)
    value = torch.zeros(horizon - 1, batch_size, 1)
    discount = torch.ones(horizon - 1, batch_size, 1)
    bootstrap = torch.zeros(batch_size, 1)
    lambda_ = 1
    returns = lambda_return(reward, value, discount, bootstrap, lambda_)
    assert returns.shape == (horizon - 1, batch_size, 1)
    assert (returns != reward).any()
