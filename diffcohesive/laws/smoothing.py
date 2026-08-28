"""Smooth replacements for the non-differentiable operators cohesive laws rely on."""

import torch


def smooth_macaulay(x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """C^1 approximation of <x> = max(x, 0), exact as eps -> 0."""
    return 0.5 * (x + torch.sqrt(x * x + eps * eps))


def smooth_max(a: torch.Tensor, b: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """C^1 approximation of max(a, b), used to advance the history variable kappa."""
    return 0.5 * (a + b + torch.sqrt((a - b) * (a - b) + eps * eps))
