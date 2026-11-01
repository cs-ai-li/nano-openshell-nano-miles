import math
import pytest
import torch

from nano_miles.async_train import run_async_grpo_training
from nano_miles.rollout import SGLangClient
from nano_miles.train import GRPOConfig
from nano_miles.trainer import PolicyTrainer


@pytest.fixture(scope="module")
def client():
    return SGLangClient(base_url="http://127.0.0.1:30000", model_path="Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture(scope="module")
def trainer():
    return PolicyTrainer(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        lr=2e-5,
        device="cuda:0",
    )


def test_fully_async_grpo_convergence_and_staleness(client, trainer):
    """
    Stage 6 Contract: Decoupled Fully Asynchronous GRPO Closed Loop Verification.
    Runs 6 training steps with a background rollout worker concurrently producing samples,
    verifies staleness filtering, pause/resume coordination, and mathematical convergence.
    """
    dataset = [
        ("Calculate (15 * 4) + 27. Put only the final integer in <answer>...</answer>.", "87"),
        ("Calculate 100 - (12 * 6). Put only the final integer in <answer>...</answer>.", "28"),
        ("Calculate (8 * 9) - 35. Put only the final integer in <answer>...</answer>.", "37"),
        ("Calculate (25 * 3) + 45. Put only the final integer in <answer>...</answer>.", "120"),
    ]

    initial_weights = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}

    config = GRPOConfig(
        sglang_url=client.base_url,
        lr=2e-5,
        group_size=6,
        ppo_epochs=2,
        kl_coef=0.0,
        temperature=0.8,
        max_new_tokens=64,
        format_bonus=0.1,
        accuracy_bonus=1.0,
    )

    try:
        history = run_async_grpo_training(
            dataset=dataset,
            config=config,
            num_steps=6,
            batch_prompts=2,
            max_staleness=2,
            buffer_size=4,
            trainer=trainer,
            client=client,
        )

        assert len(history) == 6, f"Expected 6 async steps, got {len(history)}"

        # 1. Health & Numerical Stability Verification
        for h in history:
            assert not math.isnan(h["reward_mean"]) and not math.isinf(h["reward_mean"])
            assert not math.isnan(h["pg_loss"]) and not math.isinf(h["pg_loss"])
            assert h["staleness"] <= 2, f"Batch staleness {h['staleness']} exceeded max_staleness 2"
            assert h["ipc_sync_ms"] < 1000.0, f"IPC sync exceeded 1000ms: {h['ipc_sync_ms']}ms"

        early_reward = sum(h["reward_mean"] for h in history[:2]) / 2.0
        late_reward = sum(h["reward_mean"] for h in history[-2:]) / 2.0
        max_acc = max(h["acc_rate"] for h in history)
        avg_staleness = sum(h["staleness"] for h in history) / len(history)
        max_staleness_observed = max(h["staleness"] for h in history)

        print("\n" + "=" * 60)
        print("[Stage 6 Async Training Results]")
        print(f"Early Mean Reward (Steps 1-2): {early_reward:.4f}")
        print(f"Late Mean Reward (Steps 5-6):  {late_reward:.4f}")
        print(f"Max Accuracy Achieved:         {max_acc * 100:.1f}%")
        print(f"Average Staleness:             {avg_staleness:.2f}")
        print(f"Max Staleness Observed:        {max_staleness_observed}")
        print(f"Reward Improvement (Delta):    {late_reward - early_reward:+.4f}")
        print("=" * 60)

        # 2. Convergence Assertion: Late reward should improve or stay high
        assert late_reward >= early_reward or late_reward >= 0.3, (
            f"Expected async RL reward progression: late {late_reward:.3f} vs early {early_reward:.3f}"
        )
        assert max_acc >= 0.15, f"Expected max accuracy >= 15%, got {max_acc * 100:.1f}%"

    finally:
        # Restore initial weights
        with torch.no_grad():
            trainer.model.load_state_dict(initial_weights)
        trainer.sync_to_sglang(sglang_url=client.base_url)
        print("Initial baseline model weights restored to SGLang successfully.")
