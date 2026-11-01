import pytest
import torch
from nano_miles.rollout import SGLangClient, build_rollout_batch
from nano_miles.schema import Trajectory, RolloutBatch


@pytest.fixture(scope="module")
def client():
    # Connects to the local SGLang instance running on RunPod
    return SGLangClient(base_url="http://127.0.0.1:30000", model_path="Qwen/Qwen2.5-0.5B-Instruct")


def test_tito_single_sample(client):
    """
    Stage 1 Contract 1: Token-in-Token-Out sampling verification.
    Ensures input_ids are fed directly, output_token_logprobs are extracted,
    and tokens are strictly concatenated without re-tokenization.
    """
    prompt = "Question: What is 2 + 3? Answer with just the number."
    traj = client.generate_sync(prompt, sampling_params={"temperature": 0.0, "max_new_tokens": 16})

    # 1. Structure assertions
    assert isinstance(traj, Trajectory)
    assert traj.prompt == prompt
    assert len(traj.prompt_tokens) > 0
    assert len(traj.response_tokens) > 0

    # 2. Strict concatenation contract: tokens == prompt_tokens + response_tokens
    assert traj.tokens == traj.prompt_tokens + traj.response_tokens
    assert traj.total_len == traj.prompt_len + traj.response_len

    # 3. Logprob alignment contract: len(rollout_log_probs) == len(response_tokens)
    assert len(traj.rollout_log_probs) == len(traj.response_tokens)
    for lp in traj.rollout_log_probs:
        assert isinstance(lp, float)
        assert lp <= 0.0, f"log_prob {lp} should be non-positive"

    # 4. Text preview
    decoded_text = client.tokenizer.decode(traj.tokens)
    print()
    print(f"[Test 1] Generated response text: {traj.response_text!r}")
    print(f"[Test 1] Full sequence decoded text: {decoded_text!r}")
    print(f"[Test 1] Prompt tokens: {traj.prompt_len}, Response tokens: {traj.response_len}")


def test_prefill_scoring_parity(client):
    """
    Stage 1 Contract 2: Prefill Logprobs Parity Check.
    Validates that score_prefill_logprobs() computes logprobs matching the
    rollout decode logprobs across the same sequence (prefill vs decode parity).
    """
    prompt = "The capital of France is Paris. The capital of Germany is"
    traj = client.generate_sync(prompt, sampling_params={"temperature": 0.0, "max_new_tokens": 8})

    prefill_logprobs = client.score_prefill_logprobs_sync(
        tokens=traj.tokens,
        prompt_len=traj.prompt_len,
    )

    assert len(prefill_logprobs) == len(traj.rollout_log_probs)

    diffs = [abs(a - b) for a, b in zip(traj.rollout_log_probs, prefill_logprobs)]
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)

    print()
    print(f"[Test 2] Rollout decode logprobs: {traj.rollout_log_probs[:5]}")
    print(f"[Test 2] Prefill scored logprobs: {prefill_logprobs[:5]}")
    print(f"[Test 2] Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    # Floating point decode vs prefill kernel minor differences, max diff typically < 0.1
    assert max_diff < 0.1, f"Prefill logprobs diverged too much from rollout logprobs: max_diff={max_diff}"


def test_batch_assembly_and_shift_by_one_alignment(client):
    """
    Stage 1 Contract 3: RolloutBatch collation & Shift-by-One alignment verification.
    """
    prompts = [
        "Count: 1, 2, 3,",
        "Explain what Python is in five words.",
        "What is the color of an emerald stone?",
    ]

    trajectories = [
        client.generate_sync(p, sampling_params={"temperature": 0.5, "max_new_tokens": 24})
        for p in prompts
    ]

    batch = build_rollout_batch(trajectories, pad_token_id=client.tokenizer.pad_token_id)

    B = len(trajectories)
    L = max(t.total_len for t in trajectories)

    # 1. Tensor dimensions
    assert batch.tokens.shape == (B, L)
    assert batch.loss_mask.shape == (B, L - 1)
    assert batch.rollout_log_probs.shape == (B, L - 1)

    # 2. Strict Shift-by-One alignment checks
    total_expected_response_tokens = 0
    for i, traj in enumerate(trajectories):
        P = traj.prompt_len
        R = traj.response_len
        total_expected_response_tokens += R

        # Prefix prompt must NOT be in loss_mask
        if P > 1:
            assert not batch.loss_mask[i, :P - 1].any(), f"Sample {i} prompt tokens leaked into loss_mask"

        # Response slice [P - 1 : P + R - 1] must be strictly True
        assert batch.loss_mask[i, P - 1 : P + R - 1].all(), f"Sample {i} response slice not fully True in loss_mask"

        # Padding tokens must NOT be in loss_mask
        if P + R - 1 < L - 1:
            assert not batch.loss_mask[i, P + R - 1:].any(), f"Sample {i} padding tokens leaked into loss_mask"

        # rollout_log_probs values must match trajectory
        extracted_lps = batch.rollout_log_probs[i, P - 1 : P + R - 1]
        expected_lps = torch.tensor(traj.rollout_log_probs, dtype=torch.float32)
        assert torch.allclose(extracted_lps, expected_lps, atol=1e-5), f"Sample {i} rollout_log_probs mismatch"

    # 3. Invariant: loss_mask.sum() == sum(response_tokens)
    assert batch.loss_mask.sum().item() == total_expected_response_tokens
    print()
    print(f"[Test 3] Batch size: {B}, Max seq len: {L}")
    print(f"[Test 3] Total active response tokens: {batch.loss_mask.sum().item()}")
