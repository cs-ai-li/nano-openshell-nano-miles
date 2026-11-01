from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from nano_miles.agentic_rollout import AgenticRolloutClient, build_agentic_rollout_batch
from nano_miles.grpo import compute_grpo_advantages
from nano_miles.openshell_env import OpenShellEnv, SandboxConfig, SupervisorContext
from nano_miles.rollout import SGLangClient
from nano_miles.schema import RolloutBatch, Trajectory
from nano_miles.trainer import PolicyTrainer, compute_grpo_policy_loss


@dataclass
class AgenticTask:
    task_id: str
    prompt: str
    ground_truth: str
    description: str


ONE_SHOT_PREFIX = (
    "You have access to an isolated bash shell. Execute commands with <bash>command</bash> "
    "and submit your final answer with <finish>answer</finish>.\n\n"
    "Example:\n"
    "Task: Check current user.\n"
    "<bash>whoami</bash>\n"
    "<tool_response>\nroot\n[exit_code: 0]\n</tool_response>\n"
    "<finish>root</finish>\n\n"
)

# 4 Real-world Shell Exploration Tasks
DEFAULT_AGENTIC_TASKS = [
    AgenticTask(
        task_id="task_sha256",
        prompt=(
            ONE_SHOT_PREFIX
            + "Task: Calculate the sha256 hash of the string 'nano_miles_rl' (without quotes, no newline) using bash. "
            "Put only the 64-character lowercase hex string in <finish>...</finish>."
        ),
        ground_truth="074bc933f4fa47cfb6fa72d1ecba6f98aeae3c3b0222a00965d1d64478345a99",
        description="Compute SHA256 checksum of a specific string",
    ),
    AgenticTask(
        task_id="task_file_lines",
        prompt=(
            ONE_SHOT_PREFIX
            + "Task: Create a file named 'numbers.txt' containing the integers from 1 to 5 (each on its own line), "
            "then count the number of lines in 'numbers.txt'. Put the integer count in <finish>...</finish>."
        ),
        ground_truth="5",
        description="Create file with 5 lines and count line total",
    ),
    AgenticTask(
        task_id="task_kernel_uname",
        prompt=(
            ONE_SHOT_PREFIX
            + "Task: Run 'uname -s' to probe the operating system kernel name. "
            "Put the lowercase name in <finish>...</finish>."
        ),
        ground_truth="linux",
        description="Inspect OS kernel name via uname",
    ),
    AgenticTask(
        task_id="task_bash_arithmetic",
        prompt=(
            ONE_SHOT_PREFIX
            + "Task: Use python3 inside bash to compute 48 * 25. "
            "Put the resulting integer in <finish>...</finish>."
        ),
        ground_truth="1200",
        description="Compute arithmetic product using python3 sub-command",
    ),
]


def evaluate_agentic_answer(candidate: str, ground_truth: str) -> float:
    """Evaluate candidate answer against ground truth with normalization."""
    cand = candidate.strip().lower()
    gt = ground_truth.strip().lower()

    if cand == gt:
        return 1.0

    # Substring matching for words/hashes (e.g. "linux", 64-char sha256)
    if len(gt) >= 4 and gt in cand:
        return 1.0

    # Numeric matching
    cand_nums = re.findall(r"[-+]?\b\d+(?:\.\d+)?\b", cand)
    gt_nums = re.findall(r"[-+]?\b\d+(?:\.\d+)?\b", gt)
    if cand_nums and gt_nums:
        if cand_nums[-1] == gt_nums[-1]:
            return 1.0

    return 0.0


@dataclass
class AgenticTrainConfig:
    model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    sglang_url: str = "http://127.0.0.1:30000"
    lr: float = 2e-5
    weight_decay: float = 0.01
    clip_grad_norm: float = 1.0
    clip_eps: float = 0.2
    kl_coef: float = 0.0
    group_size: int = 4
    ppo_epochs: int = 2
    temperature: float = 0.8
    max_turns: int = 3
    device: str = "cuda:0"
    infer_device: Optional[str] = None
    openshell_dir: str = "nano-openshell"
    sock_path: str = "/tmp/nano-supervisor.sock"


def run_agentic_training(
    tasks: Optional[List[AgenticTask]] = None,
    config: Optional[AgenticTrainConfig] = None,
    num_steps: int = 8,
    batch_prompts: int = 2,
    sync_weights: bool = True,
) -> List[Dict[str, Any]]:
    """
    True End-to-End Agentic RL Training Loop:
    Multi-turn Rollout -> Interleaved Loss Mask -> GRPO Whitening -> PPO Policy Loss -> CUDA IPC Hot Sync.
    Supports Dual-GPU decoupled training (GPU 0) and inference (GPU 1).
    """
    cfg = config or AgenticTrainConfig()
    task_pool = tasks or DEFAULT_AGENTIC_TASKS
    train_dev = cfg.device
    infer_dev = cfg.infer_device or ("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else train_dev)

    print("\n" + "=" * 100)
    print(f"nano-miles Agentic RL Closed-Loop Training | Steps: {num_steps} | Group Size: {cfg.group_size}")
    print(f"Model: {cfg.model_path} | Train Dev: {train_dev} | Infer Dev: {infer_dev} | Sync Weights: {sync_weights}")
    print("=" * 100)

    # 1. Initialize SGLang client & PyTorch Policy Trainer
    sgl_client = SGLangClient(base_url=cfg.sglang_url, model_path=cfg.model_path)
    trainer = PolicyTrainer(
        model_path=cfg.model_path,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        clip_grad_norm=cfg.clip_grad_norm,
        device=train_dev,
    )
    agent_client = AgenticRolloutClient(
        sglang_client=sgl_client,
        max_turns=cfg.max_turns,
        default_stop_sequences=["</bash>", "</finish>"],
    )

    sb_config = SandboxConfig(
        openshell_dir=cfg.openshell_dir,
        sock_path=cfg.sock_path,
        timeout_s=5.0,
        max_output_chars=2000,
    )

    if sync_weights:
        trainer.sync_to_sglang(sglang_url=cfg.sglang_url, target_device=infer_dev)

    history: List[Dict[str, Any]] = []

    # 2. Ensure Supervisor is running for out-of-process isolation
    with SupervisorContext(openshell_dir=cfg.openshell_dir, sock_path=cfg.sock_path):
        for step in range(1, num_steps + 1):
            t_step_start = time.time()

            # Select tasks for current step
            current_tasks = [
                task_pool[((step - 1) * batch_prompts + i) % len(task_pool)]
                for i in range(batch_prompts)
            ]

            # Construct environment instances for parallel group rollout
            envs: List[OpenShellEnv] = []
            for t_idx, task in enumerate(current_tasks):
                for g in range(cfg.group_size):
                    session_id = f"step{step}_t{t_idx}_g{g}_{int(time.time()*1000)%100000}"
                    env = OpenShellEnv(
                        session_id=session_id,
                        task_prompt=task.prompt,
                        ground_truth=task.ground_truth,
                        config=sb_config,
                        max_turns=cfg.max_turns,
                        reward_fn=lambda cand, gt, meta: evaluate_agentic_answer(cand, gt),
                    )
                    envs.append(env)

            # 3. Asynchronously perform Multi-Turn Agentic Rollouts
            t_rollout_start = time.time()
            trajs = asyncio.run(agent_client.rollout_group(envs))
            rollout_sec = time.time() - t_rollout_start

            # Clean up env workspaces
            for env in envs:
                env.close()

            # 4. Collate trajectories with interleaved loss mask
            batch = build_agentic_rollout_batch(trajs, pad_token_id=sgl_client.tokenizer.pad_token_id)

            # 5. Group relative advantage whitening with zero-variance guard
            batch = compute_grpo_advantages(batch, group_size=cfg.group_size)

            # 6. PPO Dual-Clip Policy Gradient Optimization
            step_metrics = {}
            parity_diff = 0.0

            for epoch in range(cfg.ppo_epochs):
                trainer.model.train()
                trainer.optimizer.zero_grad()
                train_lps = trainer.compute_log_probs(batch.tokens, batch.loss_mask)

                if epoch == 0:
                    with torch.no_grad():
                        mask_f = batch.loss_mask.float().to(trainer.device)
                        diff = (train_lps - batch.rollout_log_probs.to(trainer.device)).abs() * mask_f
                        parity_diff = (diff.sum() / mask_f.sum().clamp(min=1.0)).item()

                loss, ep_metrics = compute_grpo_policy_loss(
                    train_log_probs=train_lps,
                    rollout_log_probs=batch.rollout_log_probs.to(trainer.device),
                    advantages=batch.advantages.to(trainer.device),
                    loss_mask=batch.loss_mask.to(trainer.device),
                    clip_eps=cfg.clip_eps,
                    kl_coef=cfg.kl_coef,
                )
                loss.backward()

                if trainer.clip_grad_norm > 0:
                    gn = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.clip_grad_norm)
                    ep_metrics["grad_norm"] = float(gn.item())

                trainer.optimizer.step()
                del loss, train_lps
                torch.cuda.empty_cache()
                step_metrics = ep_metrics

            # 7. Zero-disk CUDA IPC weight hot update to SGLang server
            ipc_ms = 0.0
            if sync_weights:
                sync_res = trainer.sync_to_sglang(sglang_url=cfg.sglang_url, target_device=infer_dev)
                ipc_ms = sync_res["latency_seconds"] * 1000.0

            torch.cuda.empty_cache()

            step_sec = time.time() - t_step_start
            vram_gpu0 = torch.cuda.memory_allocated(0) / (1024 ** 3)
            vram_gpu1 = (torch.cuda.memory_allocated(1) / (1024 ** 3)) if torch.cuda.device_count() >= 2 else 0.0
            r_mean = sum(t.reward for t in trajs) / len(trajs)
            acc_rate = sum(1 for t in trajs if t.reward >= 1.0) / len(trajs)

            log_entry = {
                "step": step,
                "reward_mean": r_mean,
                "acc_rate": acc_rate,
                "pg_loss": step_metrics.get("pg_loss", 0.0),
                "clip_frac": step_metrics.get("clip_fraction", 0.0),
                "grad_norm": step_metrics.get("grad_norm", 0.0),
                "parity_diff": parity_diff,
                "ipc_ms": ipc_ms,
                "rollout_sec": rollout_sec,
                "step_sec": step_sec,
                "vram_gpu0_gb": vram_gpu0,
                "vram_gpu1_gb": vram_gpu1,
            }
            history.append(log_entry)

            vram_str = f"VRAM: [GPU0: {vram_gpu0:.2f}GB, GPU1: {vram_gpu1:.2f}GB]" if torch.cuda.device_count() >= 2 else f"VRAM: {vram_gpu0:.2f}GB"
            print(
                f"[Step {step:02d}/{num_steps:02d}] "
                f"Reward: {r_mean:+.3f} | Acc: {acc_rate*100:4.1f}% | "
                f"PG Loss: {log_entry['pg_loss']:+.4f} | Clip: {log_entry['clip_frac']:.2f} | "
                f"Parity: {parity_diff:.4f} | IPC: {ipc_ms:4.1f}ms | {vram_str} | Time: {step_sec:.1f}s"
            )

    print("=" * 100 + "\n")
    return history


if __name__ == "__main__":
    run_agentic_training()
