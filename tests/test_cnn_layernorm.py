import pytest
import torch
import torch.nn.functional as F

from rl_games.algos_torch.model_builder import ModelBuilder
from rl_games.algos_torch.torch_ext import LayerNorm2d


@pytest.mark.parametrize('shape', [(1, 3, 1, 1), (2, 4, 3, 5), (3, 1, 2, 2), (1, 1, 1, 1)])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_cnn_layernorm_matches_channelwise_reference_and_gradient(shape, dtype):
    torch.manual_seed(42)
    layer = LayerNorm2d(shape[1]).to(dtype)
    layer.gamma.copy_(torch.randn_like(layer.gamma))
    layer.beta.copy_(torch.randn_like(layer.beta))
    value = torch.randn(shape, dtype=dtype, requires_grad=True)
    reference_input = value.detach().clone().requires_grad_()
    actual = layer(value)
    expected = F.layer_norm(
        reference_input.movedim(1, -1), (shape[1],),
        layer.gamma.flatten(), layer.beta.flatten(), layer.eps,
    ).movedim(-1, 1)
    torch.testing.assert_close(actual, expected)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(value.grad, reference_input.grad, rtol=1e-4, atol=1e-5)


def test_half_precision_channel_variance_does_not_overflow():
    layer = LayerNorm2d(3).half()
    value = torch.tensor([[[[1000.]], [[-1000.]], [[0.]]]], dtype=torch.float16)
    expected = torch.tensor([[[[1 + (1.5 ** 0.5)]], [[1 - (1.5 ** 0.5)]], [[1.]]]],
                            dtype=torch.float16)
    torch.testing.assert_close(layer(value), expected)


@pytest.mark.parametrize('training', [False, True])
def test_cnn_layernorm_does_not_depend_on_other_batch_observations(training):
    torch.manual_seed(1)
    layer = LayerNorm2d(3).train(training)
    observation = torch.randn(1, 3, 4, 4)
    other = torch.randn(2, 3, 4, 4) * 100 + 200
    torch.testing.assert_close(layer(observation), layer(torch.cat([observation, other]))[:1])


@pytest.mark.parametrize('spatial_size', [1, 4])
@pytest.mark.parametrize('separate', [False, True])
def test_cnn_policy_single_observation_matches_batched_inference(spatial_size, separate):
    torch.manual_seed(42)
    params = {
        'model': {'name': 'discrete_a2c'},
        'network': {
            'name': 'actor_critic', 'separate': separate, 'normalization': 'layer_norm',
            'space': {'discrete': {}},
            'cnn': {
                'type': 'conv2d', 'activation': 'elu', 'initializer': {'name': 'default'},
                'convs': [{'filters': 4, 'kernel_size': 1, 'strides': 1, 'padding': 0}],
            },
            'mlp': {'units': [8], 'activation': 'elu', 'initializer': {'name': 'default'}},
        },
    }
    model = ModelBuilder().load(params).build({
        'actions_num': 3, 'input_shape': (spatial_size, spatial_size, 2),
        'num_seqs': 1, 'value_size': 1, 'normalize_value': False, 'normalize_input': False,
    }).eval()
    observation = torch.randn(1, spatial_size, spatial_size, 2)
    other = torch.randn_like(observation) * 100 + 200
    single = model({'is_train': False, 'obs': observation})
    batched = model({'is_train': False, 'obs': torch.cat([observation, other])})
    for name in ('logits', 'values'):
        assert torch.isfinite(single[name]).all()
        torch.testing.assert_close(single[name], batched[name][:1], rtol=1e-4, atol=1e-5)


def test_existing_cnn_layernorm_state_dict_loads_strictly():
    layer = LayerNorm2d(3)
    state = {'gamma': torch.full((3, 1, 1), 2.), 'beta': torch.full((3, 1, 1), -1.)}
    layer.load_state_dict(state, strict=True)
    assert set(layer.state_dict()) == set(state)
    for name in state:
        torch.testing.assert_close(layer.state_dict()[name], state[name])
