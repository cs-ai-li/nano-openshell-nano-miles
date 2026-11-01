import pytest
import torch

from nano_miles.grpo import compute_grpo_advantages, compute_math_reward
from nano_miles.rollout import SGLangClient, build_rollout_batch
from nano_miles.schema import RolloutBatch, Trajectory


@pytest.fixture(scope="module")
def client():
    return SGLangClient(base_url="http://127.0.0.1:30000", model_path="Qwen/Qwen2.5-0.5B-Instruct")


def test_math_reward_extractor():
    """
    Stage 2 Contract 1: Deterministic two-tier math reward verification.
    Format bonus = 0.1, Accuracy bonus = 1.0.
    """
    gt = "42"

    # 1. Valid XML format + correct answer
    r1 = compute_math_reward("Step 1: Compute.\n<answer>42</answer>", gt)
    assert abs(r1 - 1.1) < 1e-5, f"Expected 1.1, got {r1}"

    # 2. Valid XML format + wrong answer
    r2 = compute_math_reward("<answer>10</answer>", gt)
    assert abs(r2 - 0.1) < 1e-5, f"Expected 0.1, got {r2}"

    # 3. Valid hash format + correct float
    r3 = compute_math_reward("The answer is:\n#### 42.0", gt)
    assert abs(r3 - 1.1) < 1e-5, f"Expected 1.1, got {r3}"

    # 4. Valid boxed format + correct formatted number with commas
    r4 = compute_math_reward(r"Therefore, \boxed{1,000}", "1000")
    assert abs(r4 - 1.1) < 1e-5, f"Expected 1.1, got {r4}"

    # 5. No format + correct answer found at end
    r5 = compute_math_reward("The final total is 42.", gt)
    assert abs(r5 - 1.0) < 1e-5, f"Expected 1.0, got {r5}"

    # 6. No format + wrong answer / gibberish
    r6 = compute_math_reward("I have no idea what the answer is.", gt)
    assert abs(r6 - 0.0) < 1e-5, f"Expected 0.0, got {r6}"

    print("\n[Test 1] Math reward extractor passed all cases successfully!")


def test_grpo_vectorized_whitening_and_edge_cases():
    """
    Stage 2 Contract 2: Vectorized GRPO Advantage Whitening & Zero-Variance Degradation Protection.
    Constructs a Mock Batch with 2 Groups (N=2, G=4, B=8).
    - Group 0: rewards [1.0, 1.0, 0.0, 0.0] -> mean=0.5, std=0.5 -> adv=[1.0, 1.0, -1.0, -1.0]
    - Group 1: rewards [0.0, 0.0, 0.0, 0.0] -> all zero degradation -> adv=[0.0, 0.0, 0.0, 0.0]
    """
    trajectories = []
    # Group 0 rewards: [1.0, 1.0, 0.0, 0.0]
    g0_rewards = [1.0, 1.0, 0.0, 0.0]
    for r in g0_rewards:
        traj = Trajectory(
            prompt="Prompt 0",
            prompt_tokens=[10, 11],
            response_text=f"Resp {r}",
            response_tokens=[20, 21, 22],
            tokens=[10, 11, 20, 21, 22],
            rollout_log_probs=[-0.1, -0.2, -0.3],
            reward=r,
        )
        trajectories.append(traj)

    # Group 1 rewards: [0.0, 0.0, 0.0, 0.0] (All-zero edge case)
    g1_rewards = [0.0, 0.0, 0.0, 0.0]
    for r in g1_rewards:
        traj = Trajectory(
            prompt="Prompt 1",
            prompt_tokens=[10, 11, 12],
            response_text=f"Resp {r}",
            response_tokens=[30, 31],
            tokens=[10, 11, 12, 30, 31],
            rollout_log_probs=[-0.5, -0.6],
            reward=r,
        )
        trajectories.append(traj)

    batch = build_rollout_batch(trajectories, pad_token_id=0)
    batch = compute_grpo_advantages(batch, group_size=4)

    assert batch.advantages is not None
    assert batch.advantages.shape == batch.loss_mask.shape

    # Verification for Group 0:
    # Samples 0 & 1 have reward 1.0 -> adv should be +1.0 at active positions
    for i in [0, 1]:
        P, R = trajectories[i].prompt_len, trajectories[i].response_len
        active_adv = batch.advantages[i, P - 1 : P + R - 1]
        expected_adv = torch.full((R,), 1.0)
        assert torch.allclose(active_adv, expected_adv, atol=1e-4), f"Sample {i} advantage mismatch"

    # Samples 2 & 3 have reward 0.0 -> adv should be -1.0 at active positions
    for i in [2, 3]:
        P, R = trajectories[i].prompt_len, trajectories[i].response_len
        active_adv = batch.advantages[i, P - 1 : P + R - 1]
        expected_adv = torch.full((R,), -1.0)
        assert torch.allclose(active_adv, expected_adv, atol=1e-4), f"Sample {i} advantage mismatch"

    # Verification for Group 1 (All zero degradation):
    # Every sample must have advantages strictly equal to 0.0
    for i in [4, 5, 6, 7]:
        P, R = trajectories[i].prompt_len, trajectories[i].response_len
        active_adv = batch.advantages[i, P - 1 : P + R - 1]
        assert (active_adv == 0.0).all(), f"Group 1 sample {i} did not degrade to zero advantage!"

    # Universal Verification: prompt and padding positions MUST be strictly 0.0
    non_active_positions = ~batch.loss_mask
    assert (batch.advantages[non_active_positions] == 0.0).all(), "Non-active tokens leaked non-zero advantages!"

    # Trajectory advantages updated check
    for i, traj in enumerate(batch.trajectories):
        assert traj.advantages is not None
        assert len(traj.advantages) == traj.response_len

    print("\n[Test 2] Vectorized GRPO advantage computation and zero-variance protection passed!")


def test_end_to_end_group_rollout_and_advantage(client):
    """
    Stage 2 Contract 3: Real end-to-end SGLang group sampling + reward + advantage calculation.
    """
    prompts = [
        "What is 17 + 28? Think step by step and put the final number in <answer>...</answer>.",
        "What is 9 * 8? Think step by step and put the final number in <answer>...</answer>.",
    ]
    ground_truths = ["45", "72"]
    group_size = 4

    # 1. Concurrent group generation (N=2, G=4 -> B=8)
    trajectories = client.generate_group_sync(
        prompts,
        group_size=group_size,
        sampling_params={"temperature": 0.8, "max_new_tokens": 128},
    )
    assert len(trajectories) == 8

    # 2. Assign rewards
    for idx, traj in enumerate(trajectories):
        prompt_idx = idx // group_size
        gt = ground_truths[prompt_idx]
        traj.reward = compute_math_reward(traj.response_text, gt)

    # 3. Collate batch and compute advantages
    batch = build_rollout_batch(trajectories, pad_token_id=client.tokenizer.pad_token_id)
    batch = compute_grpo_advantages(batch, group_size=group_size)

    # 4. Validations
    assert batch.advantages is not None
    assert batch.advantages.shape == batch.loss_mask.shape
    assert (batch.advantages[~batch.loss_mask] == 0.0).all()

    print("\n" + "=" * 60)
    print(f"[Test 3] End-to-End Group Rollout & GRPO Advantage Summary (B={batch.batch_size}):")
    for i, traj in enumerate(trajectories):
        p_idx = i // group_size
        adv_val = traj.advantages[0] if traj.advantages else 0.0
        print(f"Sample {i} (Prompt {p_idx}): Reward={traj.reward:.2f}, Advantage={adv_val:.4f}")
        print(f"  Response: {traj.response_text.strip()[:80]!r}...")
    print("=" * 60)
