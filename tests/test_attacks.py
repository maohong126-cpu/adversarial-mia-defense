import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from adversral.attacks import batch_fgsm_attack, fgsm_attack, normalized_gradient_attack
from adversral.attacks import ThresholdMembershipInferenceAttack, membership_inference_attack
from adversral.engine import train_one_epoch


class LinearClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.view(x.size(0), -1))


def test_fgsm_attack_matches_gradient_sign_and_clips() -> None:
    model = LinearClassifier()
    inputs = torch.tensor([[0.5, 0.5], [0.1, 0.9]])
    labels = torch.tensor([0, 1])

    reference = inputs.clone().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(reference), labels)
    grad = torch.autograd.grad(loss, reference)[0]
    expected = torch.clamp(inputs + 0.25 * grad.sign(), 0.0, 1.0)

    actual = fgsm_attack(model, inputs, labels, epsilon=0.25)

    assert torch.allclose(actual, expected)
    assert actual.requires_grad is False
    assert torch.all((actual >= 0.0) & (actual <= 1.0))


def test_targeted_fgsm_moves_opposite_untargeted_direction() -> None:
    model = LinearClassifier()
    inputs = torch.tensor([[0.4, 0.6]])
    target_labels = torch.tensor([0])

    untargeted = fgsm_attack(model, inputs, target_labels, epsilon=0.1, targeted=False)
    targeted = fgsm_attack(model, inputs, target_labels, epsilon=0.1, targeted=True)

    assert torch.allclose(targeted - inputs, -(untargeted - inputs))


def test_normalized_gradient_attack_preserves_relative_gradient_magnitudes() -> None:
    model = LinearClassifier()
    inputs = torch.tensor([[0.3, 0.7], [0.2, 0.8]])
    labels = torch.tensor([0, 1])

    reference = inputs.clone().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(reference), labels)
    grad = torch.autograd.grad(loss, reference)[0]
    expected_direction = grad / grad.flatten(start_dim=1).norm(p=2, dim=1).view(-1, 1)
    expected = torch.clamp(inputs + 0.1 * expected_direction, 0.0, 1.0)

    actual = normalized_gradient_attack(model, inputs, labels, epsilon=0.1)

    assert torch.allclose(actual, expected)
    assert not torch.allclose(actual, torch.clamp(inputs + 0.1 * grad.sign(), 0.0, 1.0))


def test_batch_fgsm_attack_returns_all_examples() -> None:
    model = LinearClassifier()
    dataset = TensorDataset(torch.rand(5, 2), torch.tensor([0, 1, 0, 1, 0]))
    loader = DataLoader(dataset, batch_size=2)

    adv, labels = batch_fgsm_attack(model, loader, epsilon=0.05)

    assert adv.shape == (5, 2)
    assert labels.tolist() == [0, 1, 0, 1, 0]


def test_train_one_epoch_accepts_normalized_adversarial_training() -> None:
    model = LinearClassifier()
    dataset = TensorDataset(torch.rand(6, 2), torch.tensor([0, 1, 0, 1, 0, 1]))
    loader = DataLoader(dataset, batch_size=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss, acc = train_one_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        optimizer,
        torch.device("cpu"),
        adversarial_epsilon=0.05,
        adversarial_method="normalized",
    )

    assert loss > 0
    assert 0.0 <= acc <= 1.0


def test_membership_inference_threshold_predicts_high_confidence_members() -> None:
    attack = ThresholdMembershipInferenceAttack(score_type="confidence")
    attack.fit(
        member_scores=torch.tensor([0.91, 0.87, 0.95]),
        non_member_scores=torch.tensor([0.30, 0.45, 0.52]),
    )

    result = attack.evaluate(
        scores=torch.tensor([0.93, 0.35, 0.89, 0.40]),
        membership_labels=torch.tensor([1, 0, 1, 0]),
    )

    assert result.member_rule == "score_ge_threshold"
    assert result.predictions.tolist() == [True, False, True, False]
    assert result.accuracy == 1.0


def test_membership_inference_attack_scores_loader_targets() -> None:
    model = LinearClassifier()
    member_loader = DataLoader(
        TensorDataset(torch.tensor([[2.0, -2.0], [-2.0, 2.0]]), torch.tensor([0, 1])),
        batch_size=2,
    )
    non_member_loader = DataLoader(
        TensorDataset(torch.tensor([[0.05, 0.0], [0.0, 0.05]]), torch.tensor([0, 1])),
        batch_size=2,
    )
    target_loader = DataLoader(
        TensorDataset(torch.tensor([[2.0, -2.0], [0.0, 0.05]]), torch.tensor([0, 1])),
        batch_size=2,
    )

    result = membership_inference_attack(
        model,
        member_loader,
        non_member_loader,
        target_loader,
        score_type="confidence",
    )

    assert result.predictions.tolist() == [True, False]
    assert result.scores.shape == (2,)


def test_membership_inference_attack_with_membership_labels_returns_accuracy() -> None:
    model = LinearClassifier()
    member_loader = DataLoader(
        TensorDataset(torch.tensor([[2.0, -2.0], [-2.0, 2.0]]), torch.tensor([0, 1])),
        batch_size=2,
    )
    non_member_loader = DataLoader(
        TensorDataset(torch.tensor([[0.05, 0.0], [0.0, 0.05]]), torch.tensor([0, 1])),
        batch_size=2,
    )
    target_loader = DataLoader(
        TensorDataset(torch.tensor([[2.0, -2.0], [0.0, 0.05]]), torch.tensor([0, 1])),
        batch_size=2,
    )

    result = membership_inference_attack(
        model,
        member_loader,
        non_member_loader,
        target_loader,
        score_type="confidence",
        membership_labels=torch.tensor([True, False]),
    )

    assert result.accuracy == 1.0
    assert result.predictions.tolist() == [True, False]


def test_membership_inference_score_type_loss() -> None:
    attack = ThresholdMembershipInferenceAttack(score_type="loss")
    model = LinearClassifier()
    inputs = torch.tensor([[2.0, -2.0], [0.05, 0.0]])
    labels = torch.tensor([0, 1])

    scores = attack.score(model, inputs, labels)

    assert scores.shape == (2,)
    assert scores[0] < scores[1]  # high-confidence sample has lower loss


def test_membership_inference_score_type_entropy() -> None:
    attack = ThresholdMembershipInferenceAttack(score_type="entropy")
    model = LinearClassifier()
    inputs = torch.tensor([[2.0, -2.0], [0.05, 0.0]])

    scores = attack.score(model, inputs)

    assert scores.shape == (2,)
    assert scores[0] < scores[1]  # high-confidence sample has lower entropy
