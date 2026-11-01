from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from nano_miles.schema import RolloutBatch


def compute_grpo_policy_loss(
    train_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    clip_eps: float = 0.2,
    kl_coef: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute PPO/GRPO Dual-Clipped Loss with K3 approx KL."""
    train_lps_safe = torch.where(loss_mask, train_log_probs, rollout_log_probs)
    log_ratio = torch.clamp(train_lps_safe - rollout_log_probs, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)

    # Dual-clipped PPO surrogate loss (aligned with miles math_utils.py:263-265)
    pg1 = -ratio * advantages
    pg2 = -torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    pg_loss = torch.maximum(pg1, pg2)

    neg_log_ratio = -log_ratio
    approx_kl = torch.exp(neg_log_ratio) - neg_log_ratio - 1.0
    kl_loss = kl_coef * approx_kl
    total_token_loss = pg_loss + kl_loss

    mask_f = loss_mask.float()
    n_act = mask_f.sum().clamp(min=1.0)
    loss_red = (total_token_loss * mask_f).sum() / n_act
    pg_red = (pg_loss * mask_f).sum() / n_act
    kl_red = (kl_loss * mask_f).sum() / n_act

    clip_frac = (torch.gt(pg2, pg1).float() * mask_f).sum() / n_act
    mean_ratio = (ratio * mask_f).sum() / n_act
    mean_kl = (approx_kl * mask_f).sum() / n_act

    metrics = {
        "loss": float(loss_red.item()), "pg_loss": float(pg_red.item()),
        "kl_loss": float(kl_red.item()), "approx_kl": float(mean_kl.item()),
        "clip_fraction": float(clip_frac.item()), "mean_ratio": float(mean_ratio.item()),
    }
    return loss_red, metrics


class PolicyTrainer:
    """Single-card PyTorch Policy Trainer on cuda:0 with FP32 log-prob cross entropy."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        lr: float = 5e-6,
        weight_decay: float = 0.01,
        clip_grad_norm: float = 1.0,
        device: str = "cuda:0",
        micro_batch_size: int = 2,
    ):
        self.device = torch.device(device)
        self.clip_grad_norm = clip_grad_norm
        self.micro_batch_size = micro_batch_size
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(self.device)
        self.model.gradient_checkpointing_enable()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def compute_log_probs(self, tokens: torch.LongTensor, loss_mask: torch.BoolTensor) -> torch.FloatTensor:
        input_ids = tokens[:, :-1].to(self.device)
        target_ids = tokens[:, 1:].to(self.device)
        mask = loss_mask.to(self.device)
        B, L_1 = input_ids.shape

        lps_list = []
        for start_idx in range(0, B, self.micro_batch_size):
            end_idx = min(start_idx + self.micro_batch_size, B)
            sub_inputs = input_ids[start_idx:end_idx]
            sub_targets = target_ids[start_idx:end_idx]
            sub_logits = self.model(sub_inputs).logits

            for b in range(sub_logits.size(0)):
                # Compute cross-entropy per sequence in FP32 to avoid massive 151k vocab FP32 memory spike
                # while guaranteeing FP32 LogSumExp numerical precision.
                lps_b = -F.cross_entropy(sub_logits[b].float(), sub_targets[b], reduction="none")
                lps_list.append(lps_b)

        log_probs = torch.stack(lps_list, dim=0)
        return log_probs * mask.float()

    def train_step(self, batch: RolloutBatch, clip_eps: float = 0.2, kl_coef: float = 0.0) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        train_lps = self.compute_log_probs(batch.tokens, batch.loss_mask)
        loss, metrics = compute_grpo_policy_loss(
            train_log_probs=train_lps, rollout_log_probs=batch.rollout_log_probs.to(self.device),
            advantages=batch.advantages.to(self.device), loss_mask=batch.loss_mask.to(self.device),
            clip_eps=clip_eps, kl_coef=kl_coef,
        )
        loss.backward()
        if self.clip_grad_norm > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            metrics["grad_norm"] = float(gn.item())
        self.optimizer.step()
        return metrics

    def sync_to_sglang(
        self,
        sglang_url: str = "http://127.0.0.1:30000",
        target_device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        from nano_miles.weight_sync import sync_weights_to_sglang
        return sync_weights_to_sglang(self.model, sglang_url=sglang_url, target_device=target_device)
