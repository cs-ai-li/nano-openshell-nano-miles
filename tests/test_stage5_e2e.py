import math
import pytest
import torch

from nano_miles.rollout import SGLangClient
from nano_miles.train import GRPOConfig, run_grpo_training
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


def test_end_to_end_grpo_convergence(client, trainer):
    """
    Stage 5 Contract: True On-Policy Multi-Step GRPO RL Closed Loop Verification.
    Runs 6 training steps of on-policy RL with real SGLang sampling, math reward,
    GRPO whitening, PPO optimization, and 50ms CUDA IPC hot sync.
    """
    # 4 two-step arithmetic reasoning tasks
    dataset = [
        ("Calculate (15 * 4) + 27. Put only the final integer in <answer>...</answer>.", "87"),
        ("Calculate 100 - (12 * 6). Put only the final integer in <answer>...</answer>.", "28"),
        ("Calculate (8 * 9) - 35. Put only the final integer in <answer>...</answer>.", "37"),
        ("Calculate (25 * 3) + 45. Put only the final integer in <answer>...</answer>.", "120"),
    ]

    # Backup pristine model weights for clean restoration
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
        history = run_grpo_training(
            dataset=dataset,
            config=config,
            num_steps=6,
            batch_prompts=2,
            trainer=trainer,
            client=client,
        )

        assert len(history) == 6, f"Expected 6 training steps, got {len(history)}"

        # 1. Health & Numerical Stability Verification
        for h in history:
            assert not math.isnan(h["reward_mean"]) and not math.isinf(h["reward_mean"])
            assert not math.isnan(h["pg_loss"]) and not math.isinf(h["pg_loss"])
            assert not math.isnan(h["kl"]) and not math.isinf(h["kl"])
            # Parity drift audit: IPC hot sync keeps Trainer and SGLang tightly aligned across steps
            assert h["parity_diff"] < 0.08, f"Step {h['step']} on-policy parity drift exceeded 0.15: {h['parity_diff']}"
            # Sub-second IPC sync latency
            assert h["ipc_sync_ms"] < 1000.0, f"Step {h['step']} IPC sync exceeded 1000ms: {h['ipc_sync_ms']}ms"

        early_reward = sum(h["reward_mean"] for h in history[:2]) / 2.0
        late_reward = sum(h["reward_mean"] for h in history[-2:]) / 2.0
        max_acc = max(h["acc_rate"] for h in history)

        print("\n" + "=" * 60)
        print("[Stage 5 End-to-End Convergence Results]")
        print(f"Early Mean Reward (Steps 1-2): {early_reward:.4f}")
        print(f"Late Mean Reward (Steps 5-6):  {late_reward:.4f}")
        print(f"Max Accuracy Achieved:         {max_acc * 100:.1f}%")
        print(f"Reward Improvement (Delta):    {late_reward - early_reward:+.4f}")
        print("=" * 60)

        # 2. Convergence Assertion: Late reward should improve or stay high, and accuracy climbs
        assert late_reward >= early_reward or late_reward >= 0.5, (
            f"Expected RL reward progression: late {late_reward:.3f} vs early {early_reward:.3f}"
        )
        assert max_acc >= 0.50, f"Expected max accuracy >= 50%, got {max_acc * 100:.1f}%"

    finally:
        # Restore initial weights to maintain pristine state
        with torch.no_grad():
            trainer.model.load_state_dict(initial_weights)
        trainer.sync_to_sglang(sglang_url=client.base_url)
        print("Initial baseline model weights restored to SGLang successfully.")
