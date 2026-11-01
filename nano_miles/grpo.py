import re
from typing import List, Optional, Tuple
import torch
from nano_miles.schema import RolloutBatch, Trajectory


def extract_answer_candidate(response_text: str) -> Tuple[Optional[str], bool]:
    """Extract candidate answer and whether the response met format contracts."""
    xml_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL | re.IGNORECASE)
    if xml_match:
        return xml_match.group(1).strip(), True
    hash_match = re.search(r"####\s*([^\n]+)", response_text)
    if hash_match:
        return hash_match.group(1).strip(), True
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", response_text)
    if boxed_match:
        return boxed_match.group(1).strip(), True
    numbers = re.findall(r"[-+]?\b\d+(?:\.\d+)?\b", response_text)
    if numbers:
        return numbers[-1].strip(), False
    return None, False


def _normalize_numeric(val_str: str) -> Optional[float]:
    if not val_str:
        return None
    cleaned = val_str.replace(",", "").strip().rstrip(".")
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", cleaned)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def compute_math_reward(
    response_text: str,
    ground_truth: str,
    format_bonus: float = 0.1,
    accuracy_bonus: float = 1.0,
) -> float:
    """Two-tier math reward: format_bonus (0.1) + accuracy_bonus (1.0)."""
    candidate, has_format = extract_answer_candidate(response_text)
    reward = format_bonus if has_format else 0.0
    if candidate is not None:
        c_num, g_num = _normalize_numeric(candidate), _normalize_numeric(ground_truth)
        is_correct = (
            abs(c_num - g_num) < 1e-5
            if (c_num is not None and g_num is not None)
            else candidate.strip().lower() == ground_truth.strip().lower()
        )
        if is_correct:
            reward += accuracy_bonus
    return reward


def compute_grpo_advantages(batch: RolloutBatch, group_size: int, eps: float = 1e-6) -> RolloutBatch:
    """Vectorized GRPO Group Whitening with zero-variance protection."""
    if batch.rewards is None:
        raise ValueError("batch.rewards cannot be None")
    B = batch.batch_size
    if B % group_size != 0:
        raise ValueError(f"Batch size {B} not divisible by group_size {group_size}")
    N = B // group_size
    rewards_group = batch.rewards.view(N, group_size)
    mean = rewards_group.mean(dim=1, keepdim=True)
    std = rewards_group.std(dim=1, keepdim=True, correction=0)
    adv_scalar_group = torch.where(
        std < 1e-7, torch.zeros_like(rewards_group), (rewards_group - mean) / (std + eps)
    )
    adv_scalar = adv_scalar_group.view(B, 1)
    advantages = adv_scalar.expand_as(batch.loss_mask) * batch.loss_mask.float()
    batch.advantages = advantages
    for i, traj in enumerate(batch.trajectories):
        P, R = traj.prompt_len, traj.response_len
        traj.advantages = advantages[i, P - 1 : P + R - 1].tolist()
    return batch
