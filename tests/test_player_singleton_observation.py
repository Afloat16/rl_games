import gymnasium as gym
import pytest
import torch

from rl_games.algos_torch.players import PpoPlayerDiscrete
from rl_games.common.tr_helpers import unsqueeze_obs


@pytest.mark.parametrize('shape', [(), (1,), (4,), (2, 3), (3, 8, 8)])
def test_unbatched_observation_always_gains_one_dimension(shape):
    observation = torch.zeros(shape)
    batched = unsqueeze_obs(observation)
    assert batched.shape == (1,) + shape
    torch.testing.assert_close(batched[0], observation)


def test_singleton_and_scalar_leaves_in_nested_observations():
    observation = {'state': torch.tensor([0.2]), 'nested': {'value': torch.tensor(0.7)}}
    batched = unsqueeze_obs(observation)
    assert batched['state'].shape == (1, 1)
    assert batched['nested']['value'].shape == (1,)


@pytest.mark.parametrize('observation_size', [1, 4])
@pytest.mark.parametrize('normalize_input', [False, True])
@pytest.mark.parametrize('recurrent', [False, True])
def test_single_env_multidiscrete_player_handles_single_feature(
        observation_size, normalize_input, recurrent):
    network = {
        'name': 'actor_critic', 'separate': False,
        'space': {'multi_discrete': {}},
        'mlp': {'units': [8], 'activation': 'elu', 'initializer': {'name': 'default'}},
    }
    if recurrent:
        network['rnn'] = {'name': 'gru', 'units': 8, 'layers': 1, 'before_mlp': False}
    params = {
        'model': {'name': 'multi_discrete_a2c'},
        'network': network,
        'config': {
            'env_name': 'local_test', 'device_name': 'cpu',
            'normalize_input': normalize_input,
            'env_info': {
                'observation_space': gym.spaces.Box(-1, 1, (observation_size,)),
                'action_space': gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(4))),
            },
        },
    }
    player = PpoPlayerDiscrete(params)
    player.reset()
    observation = torch.full((observation_size,), 0.2)
    for _ in range(3):
        action = player.get_action(observation, is_deterministic=True)
        assert action.shape == (2,)
        assert player.action_space.contains(tuple(action.tolist()))
