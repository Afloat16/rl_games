import math
from types import SimpleNamespace

import pytest
import torch

from rl_games.algos_torch.model_builder import ModelBuilder
from rl_games.algos_torch.models import apply_sigma_parametrization


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('minimum,maximum', [(0.0, 0.3), (0.2, 1.25)])
def test_exponential_sigma_cap_is_finite_for_large_finite_logits(dtype, minimum, maximum):
    raw = torch.tensor([-30., 0., 10., 80., 90., 1000.], dtype=dtype, requires_grad=True)
    network = SimpleNamespace(sigma_parametrization='exp', min_sigma=minimum, max_sigma=maximum)
    sigma, logstd = apply_sigma_parametrization(raw, network)
    span = maximum - minimum
    expected = torch.tensor([
        minimum + span / (1 + span * math.exp(-float(value)))
        for value in raw.detach()
    ], dtype=sigma.dtype)
    assert torch.isfinite(sigma).all()
    assert torch.isfinite(logstd).all()
    torch.testing.assert_close(sigma, expected)
    torch.testing.assert_close(logstd, sigma.log())
    (sigma.sum() + logstd.sum()).backward()
    assert torch.isfinite(raw.grad).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('minimum,maximum', [(0.0, 0.3), (0.2, 1.25)])
def test_exponential_cap_preserves_small_finite_tail_gradients(dtype, minimum, maximum):
    raw = torch.tensor([80., 90.], dtype=dtype, requires_grad=True)
    network = SimpleNamespace(sigma_parametrization='exp', min_sigma=minimum, max_sigma=maximum)
    sigma, _ = apply_sigma_parametrization(raw, network)
    sigma.sum().backward()
    span = maximum - minimum
    reciprocal = torch.exp(-raw.detach().double())
    expected = (span ** 2 * reciprocal / (1 + span * reciprocal).square()).to(dtype)
    assert (raw.grad > 0).all()
    torch.testing.assert_close(raw.grad, expected, rtol=3e-5, atol=0)


@pytest.mark.parametrize('fixed_sigma', [False, True])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_large_exponential_head_keeps_policy_forward_and_backward_finite(fixed_sigma, dtype):
    params = {
        'model': {'name': 'continuous_a2c_logstd'},
        'network': {
            'name': 'actor_critic', 'separate': False,
            'space': {'continuous': {
                'mu_activation': 'None', 'sigma_activation': 'None',
                'mu_init': {'name': 'default'},
                'sigma_init': {'name': 'const_initializer', 'val': 100.},
                'fixed_sigma': fixed_sigma, 'sigma_parametrization': 'exp',
                'min_sigma': 0.2, 'max_sigma': 1.25,
            }},
            'mlp': {'units': [8], 'activation': 'elu', 'initializer': {'name': 'default'}},
        },
    }
    model = ModelBuilder().load(params).build({
        'actions_num': 3, 'input_shape': (5,), 'num_seqs': 1, 'value_size': 1,
        'normalize_value': False, 'normalize_input': False,
    }).to(dtype)
    if not fixed_sigma:
        with torch.no_grad():
            model.a2c_network.sigma.weight.zero_()
            model.a2c_network.sigma.bias.fill_(100.)
    output = model({
        'is_train': True, 'obs': torch.zeros(2, 5, dtype=dtype),
        'prev_actions': torch.zeros(2, 3),
    })
    for name in ('sigmas', 'entropy', 'prev_neglogp'):
        assert torch.isfinite(output[name]).all()
    (output['prev_neglogp'].sum() + output['entropy'].sum()).backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
