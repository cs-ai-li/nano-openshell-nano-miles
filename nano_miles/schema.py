from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class Trajectory:
    """Single TITO rollout trajectory with prompt and response tokens."""
    prompt: str
    prompt_tokens: List[int]
    response_text: str
    response_tokens: List[int]
    tokens: List[int]
    rollout_log_probs: List[float]
    reward: float = 0.0
    advantages: Optional[List[float]] = None
    weight_version: int = 0
    response_loss_mask: Optional[List[bool]] = None

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)

    @property
    def response_len(self) -> int:
        return len(self.response_tokens)

    @property
    def total_len(self) -> int:
        return len(self.tokens)

    def __post_init__(self):
        assert len(self.tokens) == len(self.prompt_tokens) + len(self.response_tokens)
        assert len(self.rollout_log_probs) == len(self.response_tokens)
        if self.response_loss_mask is not None:
            assert len(self.response_loss_mask) == len(self.response_tokens)


@dataclass
class RolloutBatch:
    """
    Batched trajectories with strict Shift-by-One alignment.
    Response slice in target prediction space [L - 1] is strictly [P - 1 : P + R - 1].
    """
    tokens: torch.LongTensor              # [B, L]
    loss_mask: torch.BoolTensor           # [B, L - 1]
    rollout_log_probs: torch.FloatTensor  # [B, L - 1]
    advantages: Optional[torch.FloatTensor] = None  # [B, L - 1]
    rewards: Optional[torch.FloatTensor] = None     # [B]
    trajectories: List[Trajectory] = field(default_factory=list)
    weight_version: int = 0

    @property
    def batch_size(self) -> int:
        return self.tokens.shape[0]

    @property
    def max_len(self) -> int:
        return self.tokens.shape[1]

    def to(self, device: torch.device | str) -> "RolloutBatch":
        self.tokens = self.tokens.to(device)
        self.loss_mask = self.loss_mask.to(device)
        self.rollout_log_probs = self.rollout_log_probs.to(device)
        if self.advantages is not None:
            self.advantages = self.advantages.to(device)
        if self.rewards is not None:
            self.rewards = self.rewards.to(device)
        return self
