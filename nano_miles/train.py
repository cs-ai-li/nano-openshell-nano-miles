from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
from nano_miles.grpo import compute_grpo_advantages, compute_math_reward, extract_answer_candidate
from nano_miles.rollout import SGLangClient, build_rollout_batch
from nano_miles.trainer import PolicyTrainer, compute_grpo_policy_loss


@dataclass
class GRPOConfig:
    model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    sglang_url: str = "http://127.0.0.1:30000"
    lr: float = 2e-5
    weight_decay: float = 0.01
    clip_grad_norm: float = 1.0
    clip_eps: float = 0.2
    kl_coef: float = 0.0
    group_size: int = 6
    ppo_epochs: int = 2
    temperature: float = 0.8
    max_new_tokens: int = 64
    format_bonus: float = 0.1
    accuracy_bonus: float = 1.0
    device: str = "cuda:0"


def run_grpo_training(
    dataset: List[Tuple[str, str]], config: Optional[GRPOConfig] = None,
    num_steps: int = 6, batch_prompts: int = 2,
    trainer: Optional[PolicyTrainer] = None, client: Optional[SGLangClient] = None,
) -> List[Dict[str, Any]]:
    """True On-Policy GRPO Closed-Loop Training Loop."""
    cfg = config or GRPOConfig()
    cli = client or SGLangClient(base_url=cfg.sglang_url, model_path=cfg.model_path)
    trn = trainer or PolicyTrainer(
        model_path=cfg.model_path, lr=cfg.lr, weight_decay=cfg.weight_decay,
        clip_grad_norm=cfg.clip_grad_norm, device=cfg.device
    )
    history: List[Dict[str, Any]] = []

    print("\n" + "=" * 90 + f"\nnano-miles True On-Policy RL | Steps: {num_steps} | Group: {cfg.group_size}\n" + "=" * 90)

    for step in range(1, num_steps + 1):
        items = [dataset[((step - 1) * batch_prompts + i) % len(dataset)] for i in range(batch_prompts)]
        prompts, ground_truths = [p for p, _ in items], [gt for _, gt in items]

        trajs = cli.generate_group_sync(
            prompts, group_size=cfg.group_size,
            sampling_params={"temperature": cfg.temperature, "max_new_tokens": cfg.max_new_tokens},
        )
        for i, t in enumerate(trajs):
            t.reward = compute_math_reward(
                t.response_text, ground_truths[i // cfg.group_size],
                format_bonus=cfg.format_bonus, accuracy_bonus=cfg.accuracy_bonus
            )

        batch = build_rollout_batch(trajs, pad_token_id=cli.tokenizer.pad_token_id)
        batch = compute_grpo_advantages(batch, group_size=cfg.group_size)

        parity_diff, step_metrics = 0.0, {}
        for epoch in range(cfg.ppo_epochs):
            trn.model.train()
            trn.optimizer.zero_grad()
            train_lps = trn.compute_log_probs(batch.tokens, batch.loss_mask)

            if epoch == 0:
                with torch.no_grad():
                    mask_f = batch.loss_mask.float().to(trn.device)
                    diff = (train_lps - batch.rollout_log_probs.to(trn.device)).abs() * mask_f
                    parity_diff = (diff.sum() / mask_f.sum().clamp(min=1.0)).item()

            loss, ep_metrics = compute_grpo_policy_loss(
                train_log_probs=train_lps, rollout_log_probs=batch.rollout_log_probs.to(trn.device),
                advantages=batch.advantages.to(trn.device), loss_mask=batch.loss_mask.to(trn.device),
                clip_eps=cfg.clip_eps, kl_coef=cfg.kl_coef,
            )
            loss.backward()
            if trn.clip_grad_norm > 0:
                gn = torch.nn.utils.clip_grad_norm_(trn.model.parameters(), trn.clip_grad_norm)
                ep_metrics["grad_norm"] = float(gn.item())
            trn.optimizer.step()
            step_metrics = ep_metrics

        sync_res = trn.sync_to_sglang(sglang_url=cfg.sglang_url)
        ipc_ms = sync_res["latency_seconds"] * 1000.0

        r_mean = sum(t.reward for t in trajs) / len(trajs)
        acc_rate = sum(1 for t in trajs if t.reward >= cfg.accuracy_bonus) / len(trajs)
        fmt_rate = sum(1 for t in trajs if extract_answer_candidate(t.response_text)[1]) / len(trajs)

        log_entry = {
            "step": step, "reward_mean": r_mean, "format_rate": fmt_rate, "acc_rate": acc_rate,
            "pg_loss": step_metrics.get("pg_loss", 0.0), "kl": step_metrics.get("approx_kl", 0.0),
            "clip_frac": step_metrics.get("clip_fraction", 0.0), "grad_norm": step_metrics.get("grad_norm", 0.0),
            "parity_diff": parity_diff, "ipc_sync_ms": ipc_ms,
        }
        history.append(log_entry)
        print(f"[Step {step:02d}/{num_steps:02d}] Reward: {r_mean:.3f} | Fmt: {fmt_rate*100:4.1f}% | Acc: {acc_rate*100:4.1f}% | PG: {log_entry['pg_loss']:+.4f} | Clip: {log_entry['clip_frac']:.2f} | Parity: {parity_diff:.4f} | IPC: {ipc_ms:4.1f}ms")

    print("=" * 90 + "\n")
    return history
