import pytest
import torch
from nano_miles.train_agentic import (
    DEFAULT_AGENTIC_TASKS,
    AgenticTrainConfig,
    run_agentic_training,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for E2E Agentic RL training")
def test_stage8_e2e_agentic_training_smoke():
    """
    Stage 8 Contract: Real GPU End-to-End Agentic RL Closed-Loop Smoke Test.
    Executes 2 real training steps with 4 parallel trajectories per task:
      1. OpenShell Supervisor & Sandboxes active
      2. Multi-turn Agentic Rollout via real SGLang server
      3. Interleaved Loss Mask strictly zeroes observation gradients
      4. Vectorized GRPO Advantage Whitening
      5. PPO Policy Loss backward pass
      6. Zero-disk CUDA IPC weight hot synchronization
    """
    cfg = AgenticTrainConfig(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        sglang_url="http://127.0.0.1:30000",
        group_size=4,
        ppo_epochs=1,
        max_turns=3,
        device="cuda:0",
        infer_device="cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0",
    )

    history = run_agentic_training(
        tasks=DEFAULT_AGENTIC_TASKS[:2],
        config=cfg,
        num_steps=2,
        batch_prompts=1,
        sync_weights=True,
    )

    assert len(history) == 2, f"Expected 2 steps, got {len(history)}"

    for entry in history:
        assert "reward_mean" in entry
        assert "pg_loss" in entry
        assert "ipc_ms" in entry
        assert entry["ipc_ms"] < 300.0, f"IPC sync exceeded 300ms: {entry['ipc_ms']}"
        assert entry["parity_diff"] < 0.20, f"Parity diff exceeded 0.20: {entry['parity_diff']}"
        assert entry["vram_gpu0_gb"] < 20.0, f"GPU 0 VRAM exceeded 20GB: {entry['vram_gpu0_gb']}"
