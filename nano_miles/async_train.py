import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
from nano_miles.grpo import compute_grpo_advantages, compute_math_reward
from nano_miles.rollout import SGLangClient, build_rollout_batch
from nano_miles.schema import RolloutBatch
from nano_miles.train import GRPOConfig
from nano_miles.trainer import PolicyTrainer, compute_grpo_policy_loss


class AsyncRolloutWorker:
    """Background rollout producer with zero-collision pause/resume for CUDA IPC updates."""

    def __init__(
        self, dataset: List[Tuple[str, str]], client: SGLangClient, group_size: int = 6,
        batch_prompts: int = 2, buffer_size: int = 8, temperature: float = 0.8,
        max_new_tokens: int = 64, format_bonus: float = 0.1, accuracy_bonus: float = 1.0,
    ):
        self.dataset, self.client, self.group_size = dataset, client, group_size
        self.batch_prompts, self.temperature, self.max_new_tokens = batch_prompts, temperature, max_new_tokens
        self.format_bonus, self.accuracy_bonus = format_bonus, accuracy_bonus
        self.queue: queue.Queue[RolloutBatch] = queue.Queue(maxsize=buffer_size)
        self.pause_event, self.stop_event, self.lock = threading.Event(), threading.Event(), threading.Lock()
        self.pause_event.set()
        self.current_version_ref, self._prompt_idx = [0], 0
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _worker_loop(self):
        while not self.stop_event.is_set():
            self.pause_event.wait()
            if self.stop_event.is_set():
                break
            items = [self.dataset[(self._prompt_idx + i) % len(self.dataset)] for i in range(self.batch_prompts)]
            self._prompt_idx = (self._prompt_idx + self.batch_prompts) % len(self.dataset)
            prompts, ground_truths = [p for p, _ in items], [gt for _, gt in items]
            v = self.current_version_ref[0]

            with self.lock:
                if self.stop_event.is_set() or not self.pause_event.is_set():
                    continue
                trajs = self.client.generate_group_sync(
                    prompts, group_size=self.group_size,
                    sampling_params={"temperature": self.temperature, "max_new_tokens": self.max_new_tokens},
                )

            for i, t in enumerate(trajs):
                t.weight_version = v
                t.reward = compute_math_reward(
                    t.response_text, ground_truths[i // self.group_size],
                    format_bonus=self.format_bonus, accuracy_bonus=self.accuracy_bonus
                )

            batch = build_rollout_batch(trajs, pad_token_id=self.client.tokenizer.pad_token_id)
            batch = compute_grpo_advantages(batch, group_size=self.group_size)

            while not self.stop_event.is_set():
                try:
                    self.queue.put(batch, timeout=0.2)
                    break
                except queue.Full:
                    continue

    def pause(self):
        self.pause_event.clear()
        with self.lock:
            pass

    def resume(self):
        self.pause_event.set()

    def stop(self):
        self.stop_event.set()
        self.pause_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)


def run_async_grpo_training(
    dataset: List[Tuple[str, str]], config: Optional[GRPOConfig] = None,
    num_steps: int = 6, batch_prompts: int = 2, max_staleness: int = 2,
    buffer_size: int = 6, trainer: Optional[PolicyTrainer] = None, client: Optional[SGLangClient] = None,
) -> List[Dict[str, Any]]:
    """Decoupled Asynchronous GRPO Closed-Loop Driver with staleness filtering."""
    cfg = config or GRPOConfig()
    cli = client or SGLangClient(base_url=cfg.sglang_url, model_path=cfg.model_path)
    trn = trainer or PolicyTrainer(
        model_path=cfg.model_path, lr=cfg.lr, weight_decay=cfg.weight_decay,
        clip_grad_norm=cfg.clip_grad_norm, device=cfg.device
    )
    worker = AsyncRolloutWorker(
        dataset=dataset, client=cli, group_size=cfg.group_size, batch_prompts=batch_prompts,
        buffer_size=buffer_size, temperature=cfg.temperature, max_new_tokens=cfg.max_new_tokens,
        format_bonus=cfg.format_bonus, accuracy_bonus=cfg.accuracy_bonus,
    )
    history, current_version = [], 0
    print("\n" + "=" * 96 + f"\nnano-miles Fully Async RL | Steps: {num_steps} | Max Staleness: {max_staleness}\n" + "=" * 96)

    try:
        while worker.queue.qsize() < 1:
            time.sleep(0.1)

        for step in range(1, num_steps + 1):
            while True:
                batch = worker.queue.get(timeout=30.0)
                staleness = current_version - batch.weight_version
                if staleness <= max_staleness:
                    break

            reward_mean = batch.rewards.mean().item()
            acc_rate = (batch.rewards >= cfg.accuracy_bonus).float().mean().item()
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

            worker.pause()
            sync_res = trn.sync_to_sglang(sglang_url=cfg.sglang_url)
            current_version += 1
            worker.current_version_ref[0] = current_version
            worker.resume()

            ipc_ms, q_size = sync_res["latency_seconds"] * 1000.0, worker.queue.qsize()
            log_entry = {
                "step": step, "reward_mean": reward_mean, "acc_rate": acc_rate,
                "staleness": staleness, "q_size": q_size, "version": current_version,
                "pg_loss": step_metrics.get("pg_loss", 0.0), "clip_frac": step_metrics.get("clip_fraction", 0.0),
                "parity_diff": parity_diff, "ipc_sync_ms": ipc_ms,
            }
            history.append(log_entry)
            print(f"[Step {step:02d}/{num_steps:02d}] Reward: {reward_mean:.3f} | Acc: {acc_rate*100:4.1f}% | Stale: {staleness} (v{batch.weight_version}->v{current_version}) | Queue: {q_size} | PG: {log_entry['pg_loss']:+.4f} | Clip: {log_entry['clip_frac']:.2f} | Parity: {parity_diff:.4f} | IPC: {ipc_ms:4.1f}ms")

    finally:
        worker.stop()

    print("=" * 96 + "\n")
    return history
