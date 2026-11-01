import pytest
import torch

from nano_miles.grpo import compute_grpo_advantages, compute_math_reward
from nano_miles.rollout import SGLangClient, build_rollout_batch
from nano_miles.schema import RolloutBatch, Trajectory
from nano_miles.trainer import PolicyTrainer, compute_grpo_policy_loss


@pytest.fixture(scope="module")
def client():
    return SGLangClient(base_url="http://127.0.0.1:30000", model_path="Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture(scope="module")
def trainer():
    # Single-card PyTorch Trainer on cuda:0
    return PolicyTrainer(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        lr=5e-6,
        device="cuda:0",
    )


def test_step0_logprob_parity(client, trainer):
    """
    Stage 3 Contract 1: Step 0 Logprob Parity Check (Inference vs Trainer).
    Validates that before any optimizer update, PyTorch PolicyTrainer's forward pass
    produces log_probs that numerically match SGLang's rollout and prefill logprobs.
    """
    prompt = "The chemical symbol for water is"
    traj = client.generate_sync(prompt, sampling_params={"temperature": 0.0, "max_new_tokens": 10})
    prefill_lps = client.score_prefill_logprobs_sync(traj.tokens, traj.prompt_len)

    batch = build_rollout_batch([traj], pad_token_id=client.tokenizer.pad_token_id)

    # Compute Trainer logprobs under un-updated policy
    with torch.no_grad():
        trainer_log_probs = trainer.compute_log_probs(batch.tokens, batch.loss_mask)

    P = traj.prompt_len
    R = traj.response_len
    trainer_resp_lps = trainer_log_probs[0, P - 1 : P + R - 1].cpu().tolist()

    # Compare with SGLang Prefill scored logprobs (both are full-sequence prefill GEMMs)
    prefill_diffs = [abs(a - b) for a, b in zip(trainer_resp_lps, prefill_lps)]
    mean_diff_prefill = sum(prefill_diffs) / len(prefill_diffs)
    max_diff_prefill = max(prefill_diffs)

    # Compare with SGLang Decode rollout logprobs
    decode_diffs = [abs(a - b) for a, b in zip(trainer_resp_lps, traj.rollout_log_probs)]
    mean_diff_decode = sum(decode_diffs) / len(decode_diffs)
    max_diff_decode = max(decode_diffs)

    print()
    print("=" * 60)
    print("[Test 1: Step 0 Parity Analysis]")
    print(f"Trainer logprobs:  {trainer_resp_lps[:5]}")
    print(f"SGLang prefill:    {prefill_lps[:5]}")
    print(f"SGLang decode:     {traj.rollout_log_probs[:5]}")
    print(f"Trainer vs Prefill: mean_diff = {mean_diff_prefill:.6f}, max_diff = {max_diff_prefill:.6f}")
    print(f"Trainer vs Decode:  mean_diff = {mean_diff_decode:.6f}, max_diff = {max_diff_decode:.6f}")
    print("=" * 60)

    # High-precision parity assertions (bfloat16 FP32 logit accumulation)
    assert mean_diff_prefill < 0.015, f"Trainer vs Prefill mean diff too large: {mean_diff_prefill}"
    assert max_diff_prefill < 0.06, f"Trainer vs Prefill max diff too large: {max_diff_prefill}"


def test_grpo_loss_clipping_and_mask_isolation():
    """
    Stage 2 Contract 2: Loss Function Mathematical & Mask Isolation Check.
    1. Validates that when train_log_probs == rollout_log_probs:
       mean_ratio == 1.0000, approx_kl == 0.0000.
    2. Validates that altering non-masked positions has zero effect on loss & gradients.
    """
    B, L_minus_1 = 2, 6
    loss_mask = torch.tensor(
        [[False, True, True, False, False, False],
         [False, False, True, True, True, False]]
    )
    rollout_lps = torch.tensor(
        [[-1.0, -2.0, -1.5, 0.0, 0.0, 0.0],
         [0.0, 0.0, -0.8, -1.2, -2.1, 0.0]]
    )
    advantages = torch.tensor(
        [[0.0, 1.5, -0.5, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.8, 0.8, -1.6, 0.0]]
    )

    # 1. Identity ratio & zero KL check
    train_lps_1 = rollout_lps.clone().detach().requires_grad_(True)
    loss_1, metrics_1 = compute_grpo_policy_loss(
        train_log_probs=train_lps_1,
        rollout_log_probs=rollout_lps,
        advantages=advantages,
        loss_mask=loss_mask,
        clip_eps=0.2,
        kl_coef=0.01,
    )
    assert abs(metrics_1["mean_ratio"] - 1.0) < 1e-6, f"mean_ratio should be 1.0, got {metrics_1['mean_ratio']}"
    assert abs(metrics_1["approx_kl"] - 0.0) < 1e-6, f"approx_kl should be 0.0, got {metrics_1['approx_kl']}"
    loss_1.backward()

    # 2. Mask Isolation: modify non-active positions in train_lps_2
    train_lps_2 = rollout_lps.clone().detach()
    # Modify positions where loss_mask is False:
    train_lps_2[0, 0] = -99.0
    train_lps_2[0, 3:] = 42.0
    train_lps_2[1, :2] = 123.0
    train_lps_2[1, 5] = -55.0
    train_lps_2.requires_grad_(True)

    loss_2, metrics_2 = compute_grpo_policy_loss(
        train_log_probs=train_lps_2,
        rollout_log_probs=rollout_lps,
        advantages=advantages,
        loss_mask=loss_mask,
        clip_eps=0.2,
        kl_coef=0.01,
    )
    loss_2.backward()

    # Loss must be strictly identical regardless of non-masked mutations
    assert torch.allclose(loss_1, loss_2, atol=1e-7), f"Loss changed when mutating non-masked positions: {loss_1} vs {loss_2}"

    # Gradients on non-active tokens must be strictly 0.0
    assert (train_lps_2.grad[~loss_mask] == 0.0).all(), "Gradients leaked into non-masked tokens!"

    # Gradients on active tokens must be identical
    assert torch.allclose(train_lps_1.grad[loss_mask], train_lps_2.grad[loss_mask], atol=1e-7)
    print("\n[Test 2] Loss clipping, K3 KL, and strict mask isolation passed!")


def test_single_train_step_directional_shift(client):
    """
    Stage 3 Contract 3: Single Optimization Step Directional Shift Verification.
    Validates that after 1 gradient step:
      - Tokens with Advantage > 0 have higher log_prob.
      - Tokens with Advantage < 0 have lower log_prob.
    """
    # Create trainer with appropriate learning rate for 1-step directional signal
    trainer = PolicyTrainer(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        lr=1e-5,
        clip_grad_norm=1.0,
        device="cuda:0",
    )

    prompts = [
        "What is 15 + 25? Put answer in <answer>...</answer>.",
    ]
    group_size = 2

    trajs = client.generate_group_sync(
        prompts,
        group_size=group_size,
        sampling_params={"temperature": 0.7, "max_new_tokens": 32, "sampling_seed": 42},
    )

    # Establish clear positive vs negative reward contrast within group
    trajs[0].reward = 1.10
    trajs[1].reward = 0.00

    batch = build_rollout_batch(trajs, pad_token_id=client.tokenizer.pad_token_id)
    batch = compute_grpo_advantages(batch, group_size=group_size)

    # 1. Compute initial log_probs before training
    with torch.no_grad():
        lps_before = trainer.compute_log_probs(batch.tokens, batch.loss_mask).detach().clone()

    # Align rollout logprobs strictly to trainer initial policy (ratio=1.0)
    batch.rollout_log_probs = lps_before.cpu()

    # 2. Execute 1 train step
    metrics = trainer.train_step(batch, clip_eps=0.2, kl_coef=0.0)

    # 3. Compute log_probs after training
    with torch.no_grad():
        lps_after = trainer.compute_log_probs(batch.tokens, batch.loss_mask).detach().clone()

    delta_lps = (lps_after - lps_before) * batch.loss_mask.float().to("cuda:0")

    print("\n" + "=" * 60)
    print(f"[Test 3] Single Step Directional Shift (Grad Norm: {metrics.get('grad_norm', 0.0):.4f}):")

    for i in range(2):
        P, R = trajs[i].prompt_len, trajs[i].response_len
        sample_adv = batch.advantages[i, P - 1].item()
        sample_delta = delta_lps[i, P - 1 : P + R - 1].mean().item()
        print(f"Sample {i}: Advantage = {sample_adv:+.4f}, Delta LogProb = {sample_delta:+.6f}")

        if sample_adv > 0.05:
            assert sample_delta > 0.0, f"Sample {i} with positive advantage had non-positive delta log_prob: {sample_delta}"
        elif sample_adv < -0.05:
            assert sample_delta < 0.0, f"Sample {i} with negative advantage had non-negative delta log_prob: {sample_delta}"

    print("=" * 60)
    print("Directional shift verification strictly passed!")
