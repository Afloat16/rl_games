import pytest
import torch

from rl_games.algos_torch.model_builder import ModelBuilder
from rl_games.common.divergence import d_kl_discrete
from rl_games.common.extensions.distributions import CategoricalMasked


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('mask', [[True, False, True, False],
                                 [False, True, False, False],
                                 [True, True, True, True]])
def test_masked_entropy_matches_valid_action_reference(dtype, mask):
    logits = torch.tensor([[0.7, 1.2, -0.4, 0.2], [-0.9, 0.3, 0.1, -0.1]],
                          dtype=dtype, requires_grad=True)
    mask = torch.tensor(mask)
    reference_logits = logits.detach()[:, mask].double().requires_grad_()
    expected = torch.distributions.Categorical(logits=reference_logits).entropy()
    actual = CategoricalMasked(logits=logits, masks=mask).entropy()
    actual.sum().backward()
    expected.sum().backward()

    expected_gradient = torch.zeros_like(logits, dtype=torch.float64)
    expected_gradient[:, mask] = reference_logits.grad
    tolerance = 0.03 if dtype == torch.bfloat16 else 0.005 if dtype == torch.float16 else 1e-6
    torch.testing.assert_close(actual.double(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(logits.grad.double(), expected_gradient, atol=tolerance, rtol=tolerance)
    assert actual.dtype == dtype


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32, torch.float64])
def test_masked_action_never_outweighs_low_valid_logits(dtype):
    logits = torch.tensor([[-2e8, 0.0, -3e8]], dtype=dtype)
    distribution = CategoricalMasked(logits=logits, masks=torch.tensor([[True, False, True]]))
    torch.testing.assert_close(distribution.probs, logits.new_tensor([[1.0, 0.0, 0.0]]))
    assert (distribution.sample((32,)) == 0).all()


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_masked_kl_matches_valid_action_reference(dtype):
    mask = torch.tensor([[True, False, True]])
    p = CategoricalMasked(logits=torch.tensor([[0.7, 2.0, -0.4]], dtype=dtype), masks=mask)
    q = CategoricalMasked(logits=torch.tensor([[-0.2, 0.3, 0.8]], dtype=dtype), masks=mask)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Categorical(logits=p.logits[:, mask[0]].double()),
        torch.distributions.Categorical(logits=q.logits[:, mask[0]].double()),
    )
    tolerance = 0.03 if dtype == torch.bfloat16 else 0.005 if dtype == torch.float16 else 1e-6
    torch.testing.assert_close(d_kl_discrete(p.logits, q.logits).double(), expected,
                               atol=tolerance, rtol=tolerance)


def test_zero_mass_kl_terms_and_support_loss():
    p = torch.tensor([[0.0, -torch.inf], [-torch.inf, 0.0]])
    q = torch.tensor([[0.0, -torch.inf], [0.0, -torch.inf]])
    torch.testing.assert_close(d_kl_discrete(p, q), torch.tensor([0.0, torch.inf]))


@pytest.mark.parametrize('multi_discrete', [False, True])
def test_half_precision_masked_policy_backward_stays_finite(multi_discrete):
    space = 'multi_discrete' if multi_discrete else 'discrete'
    params = {
        'model': {'name': 'multi_discrete_a2c' if multi_discrete else 'discrete_a2c'},
        'network': {
            'name': 'actor_critic', 'separate': False,
            'space': {space: {}},
            'mlp': {'units': [8], 'activation': 'elu', 'initializer': {'name': 'default'}},
        },
    }
    model = ModelBuilder().load(params).build({
        'actions_num': [3, 4] if multi_discrete else 3,
        'input_shape': (5,), 'num_seqs': 1, 'value_size': 1,
        'normalize_input': False, 'normalize_value': False,
    }).half()
    masks = [True, False, True, True, False, True, False] if multi_discrete else [True, False, True]
    output = model({
        'is_train': True, 'obs': torch.ones(2, 5, dtype=torch.float16),
        'prev_actions': torch.zeros((2, 2) if multi_discrete else (2,), dtype=torch.long),
        'action_masks': torch.tensor([masks, masks]),
    })
    loss = output['prev_neglogp'].mean() - 0.01 * output['entropy'].mean()
    assert torch.isfinite(loss)
    same_policy_kl = model.kl({'logits': output['logits']}, {'logits': output['logits']})
    torch.testing.assert_close(same_policy_kl, torch.zeros_like(same_policy_kl))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
