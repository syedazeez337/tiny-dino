"""Small action-policy candidate for tic-tac-toe."""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from .ttt_contract import FEATURE_DIM, N_ACTIONS


@dataclass(frozen=True, slots=True)
class ModelSpec:
    kind: str
    hidden: int
    layers: int


class ActionMLP(nn.Module):
    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        hidden: int = 96,
        layers: int = 3,
        out_dim: int = N_ACTIONS,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        width = in_dim
        for _ in range(layers):
            modules.extend((nn.Linear(width, hidden), nn.SiLU()))
            width = hidden
        modules.append(nn.Linear(width, out_dim))
        self.network = nn.Sequential(*modules)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.network(features))


def build_model(spec: ModelSpec) -> nn.Module:
    if spec.kind != "mlp":
        raise ValueError(f"unsupported model kind: {spec.kind}")
    return ActionMLP(FEATURE_DIM, spec.hidden, spec.layers)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def set_valued_loss(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Negative log-probability of the set of acceptable moves.

    Any acceptable move earns full credit, which matters because many positions
    have several optimal moves.
    """
    if logits.shape != allowed.shape or logits.ndim != 2:
        raise ValueError("logits and allowed must be matching [batch, actions] tensors")
    if not torch.all(allowed.any(dim=1)):
        raise ValueError("every row needs an allowed action")
    log_prob = torch.log_softmax(logits, dim=-1)
    masked = torch.where(allowed, log_prob, torch.full_like(log_prob, -torch.inf))
    return -torch.logsumexp(masked, dim=-1).mean()
