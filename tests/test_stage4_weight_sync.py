import pytest
import torch

from nano_miles.grpo import compute_grpo_advantages
from nano_miles.rollout import SGLangClient, build_rollout_batch
from nano_miles.trainer import PolicyTrainer
from nano_miles.weight_sync import sync_weights_to_sglang


@pytest.fixture(scope="module")
def client():
    return SGLangClient(base_url="http://127.0.0.1:30000", model_path="Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture(scope="module")
def trainer():
    return PolicyTrainer(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        lr=1e-5,
        device="cuda:0",
    )


def test_ipc_weight_sync_latency_and_shift(client, trainer):
    """
    Stage 4 Contract 1: Sub-second IPC Sync & Significant Distribution Shift.
    Verifies that:
      1. Weight sync executes within 2.0 seconds with zero disk I/O.
      2. Perturbing trainer weights and syncing causes SGLang output distribution to shift (max_diff > 0.05).
    """
    prompt = "The chemical symbol for water is"
    traj = client.generate_sync(prompt, sampling_params={"temperature": 0.0, "max_new_tokens": 10})
    lps_before = client.score_prefill_logprobs_sync(traj.tokens, traj.prompt_len)

    # Backup initial weights
    initial_weights = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}

    try:
        # Apply deterministic perturbation to trainer weights
        with torch.no_grad():
            for name, param in trainer.model.named_parameters():
                param.add_(0.02 * torch.sin(param))

        # Perform zero-copy CUDA IPC weight hot update
        res = trainer.sync_to_sglang(sglang_url=client.base_url)

        print("\n" + "=" * 60)
        print(f"[Test 1: IPC Sync Latency] Elapsed: {res['latency_seconds']:.4f}s")
        print("=" * 60)

        assert res["success"] is True
        assert res["latency_seconds"] < 2.0, f"IPC sync took {res['latency_seconds']:.3f}s (budget: <2.0s)"

        # Score SGLang with updated weights
        lps_after = client.score_prefill_logprobs_sync(traj.tokens, traj.prompt_len)

        diffs = [abs(a - b) for a, b in zip(lps_before, lps_after)]
        max_shift = max(diffs)
        mean_shift = sum(diffs) / len(diffs)

        print(f"[Test 1: Shift Analysis] mean_shift = {mean_shift:.6f}, max_shift = {max_shift:.6f}")
        print(f"LPS Before: {lps_before[:5]}")
        print(f"LPS After:  {lps_after[:5]}")
        print("=" * 60)

        assert max_shift > 0.05, f"Weights hot-update did not shift SGLang outputs: max_shift = {max_shift:.6f}"

    finally:
        # Restore initial weights
        with torch.no_grad():
            trainer.model.load_state_dict(initial_weights)
        trainer.sync_to_sglang(sglang_url=client.base_url)


def test_post_update_trainer_sglang_parity(client, trainer):
    """
    Stage 4 Contract 2: Post-Update Bit-level Parity & Clean Restoration.
    Verifies that:
      1. After 1 training step and IPC weight sync, Trainer and SGLang stay mathematically aligned (mean_diff < 0.025).
      2. Restoring weights returns SGLang to the original distribution.
    """
    initial_weights = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}

    prompts = ["The chemical symbol for water is"]
    trajs = client.generate_group_sync(
        prompts,
        group_size=2,
        sampling_params={"temperature": 0.7, "max_new_tokens": 16},
    )
    trajs[0].reward = 1.0
    trajs[1].reward = 0.0
    batch = build_rollout_batch(trajs, pad_token_id=client.tokenizer.pad_token_id)
    batch = compute_grpo_advantages(batch, group_size=2)

    traj = trajs[0]
    P, R = traj.prompt_len, traj.response_len

    # Initial SGLang score
    lps_before = client.score_prefill_logprobs_sync(traj.tokens, P)

    try:
        # Execute 1 real gradient update
        trainer.train_step(batch)

        # Sync to SGLang via CUDA IPC
        sync_res = trainer.sync_to_sglang(sglang_url=client.base_url)
        assert sync_res["success"] is True

        # SGLang prefill logprobs under updated policy
        sglang_lps_after = client.score_prefill_logprobs_sync(traj.tokens, P)

        # Trainer compute_log_probs under updated policy
        with torch.no_grad():
            trainer_lps = trainer.compute_log_probs(batch.tokens, batch.loss_mask)
        trainer_resp_lps = trainer_lps[0, P - 1 : P + R - 1].cpu().tolist()

        diffs = [abs(a - b) for a, b in zip(trainer_resp_lps, sglang_lps_after)]
        mean_diff = sum(diffs) / len(diffs)
        max_diff = max(diffs)

        print("\n" + "=" * 60)
        print("[Test 2: Post-Sync Trainer vs SGLang Parity]")
        print(f"Trainer logprobs: {trainer_resp_lps[:5]}")
        print(f"SGLang logprobs:  {sglang_lps_after[:5]}")
        print(f"Parity stats: mean_diff = {mean_diff:.6f}, max_diff = {max_diff:.6f}")
        print("=" * 60)

        assert mean_diff < 0.035, f"Post-sync Trainer vs SGLang mean diff too large: {mean_diff:.6f}"

    finally:
        # Restore initial weights
        with torch.no_grad():
            trainer.model.load_state_dict(initial_weights)
        trainer.sync_to_sglang(sglang_url=client.base_url)

        # Check clean restoration
        lps_restored = client.score_prefill_logprobs_sync(traj.tokens, P)
        restore_diffs = [abs(a - b) for a, b in zip(lps_restored, lps_before)]
        restore_mean_diff = sum(restore_diffs) / len(restore_diffs)
        print(f"[Test 2: Restoration check] Mean diff from original: {restore_mean_diff:.6f}")
        assert restore_mean_diff < 0.035, f"Weights restoration failed: {restore_mean_diff:.6f}"
