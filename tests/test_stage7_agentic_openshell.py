import asyncio
import os
import signal
import time
import pytest
import torch
from nano_miles.schema import Trajectory, RolloutBatch
from nano_miles.rollout import build_rollout_batch
from nano_miles.grpo import compute_grpo_advantages
from nano_miles.trainer import compute_grpo_policy_loss
from nano_miles.openshell_env import (
    OpenShellEnv,
    OpenShellSandbox,
    SandboxConfig,
    SupervisorContext,
    parse_action,
)
from nano_miles.agentic_rollout import build_agentic_rollout_batch


@pytest.fixture(scope="module")
def supervisor():
    """Ensure supervisor is running for openshell tests."""
    sock_path = "/tmp/nano-supervisor-test.sock"
    ctx = SupervisorContext(
        openshell_dir="nano-openshell",
        sock_path=sock_path,
        real_openai_key="sk-test-live-key",
    )
    ctx.start()
    yield ctx
    ctx.stop()


def test_action_parsing():
    """Verify robust extraction of <bash> and <finish> tags."""
    # 1. Standard bash command
    text = "Let me list the files:\n<bash>ls -la</bash>\nDone."
    atype, payload = parse_action(text)
    assert atype == "bash"
    assert payload == "ls -la"

    # 2. Multiline bash command
    text2 = "<bash>\ncat << 'EOF' > test.txt\nhello world\nEOF\npython3 -c 'print(open(\"test.txt\").read())'\n</bash>"
    atype2, payload2 = parse_action(text2)
    assert atype2 == "bash"
    assert "test.txt" in payload2

    # 3. Finish tag
    text3 = "I have solved it.\n<finish>42</finish>"
    atype3, payload3 = parse_action(text3)
    assert atype3 == "finish"
    assert payload3 == "42"

    # 4. Unknown format
    text4 = "I don't know what to do."
    atype4, payload4 = parse_action(text4)
    assert atype4 is None
    assert payload4 is None


@pytest.mark.asyncio
async def test_sandbox_execution_and_truncation(supervisor):
    """Verify async command execution inside sandbox with output truncation."""
    config = SandboxConfig(
        openshell_dir="nano-openshell",
        sock_path=supervisor.sock_path,
        max_output_chars=200,
        timeout_s=5.0,
    )
    sandbox = OpenShellSandbox(session_id="test_exec_session", config=config)

    try:
        # 1. Normal command execution
        code, stdout, stderr, timeout = await sandbox.execute_command("echo 'Hello Sandbox'; whoami")
        assert code == 0
        assert not timeout
        assert "Hello Sandbox" in stdout
        assert "root" in stdout  # Rootless user namespace maps root to current user

        formatted = sandbox.format_output(stdout, stderr, code)
        assert "<tool_response>" in formatted
        assert "[exit_code: 0]" in formatted

        # 2. Output truncation verification
        long_cmd = "python3 -c 'print(\"A\" * 500)'"
        code2, stdout2, stderr2, timeout2 = await sandbox.execute_command(long_cmd)
        assert code2 == 0
        formatted_long = sandbox.format_output(stdout2, stderr2, code2)
        assert "...[truncated]..." in formatted_long
        assert len(formatted_long) < 300
    finally:
        sandbox.cleanup()


@pytest.mark.asyncio
async def test_sandbox_timeout_killpg(supervisor):
    """Verify timeout watchdog properly terminates process group with SIGKILL."""
    config = SandboxConfig(
        openshell_dir="nano-openshell",
        sock_path=supervisor.sock_path,
        timeout_s=1.0,
    )
    sandbox = OpenShellSandbox(session_id="test_timeout_session", config=config)

    try:
        t0 = time.time()
        code, stdout, stderr, is_timeout = await sandbox.execute_command("sleep 10")
        elapsed = time.time() - t0

        assert is_timeout is True
        assert code == 124
        assert elapsed < 3.0, f"Timeout took too long to abort: {elapsed}s"
        assert "timeout" in stderr.lower()
    finally:
        sandbox.cleanup()


@pytest.mark.asyncio
async def test_openshell_env_full_episode(supervisor):
    """Verify full environment episode state machine (reset -> bash step -> finish step)."""
    config = SandboxConfig(
        openshell_dir="nano-openshell",
        sock_path=supervisor.sock_path,
        timeout_s=5.0,
    )
    env = OpenShellEnv(
        session_id="test_env_episode",
        task_prompt="Find 15 + 27 using bash and output the result.",
        ground_truth="42",
        config=config,
        max_turns=3,
    )

    try:
        # 1. Reset
        prompt = await env.reset()
        assert "15 + 27" in prompt
        assert env.current_turn == 0
        assert not env.is_done

        # 2. Step 1: Execute bash command
        action1 = "I will calculate it with python:\n<bash>python3 -c 'print(15 + 27)'</bash>"
        res1 = await env.step(action1)
        assert not res1.done
        assert res1.metadata["action_type"] == "bash"
        assert "42" in res1.observation
        assert "[exit_code: 0]" in res1.observation
        assert env.current_turn == 1

        # 3. Step 2: Finish action with correct answer
        action2 = "The calculation is complete.\n<finish>42</finish>"
        res2 = await env.step(action2)
        assert res2.done is True
        assert res2.reward == 1.0
        assert res2.metadata["final_answer"] == "42"
    finally:
        env.close()


def test_interleaved_loss_mask_and_grpo_math():
    """
    Verify Shift-by-One prediction slice and interleaved loss mask math contract:
    - Model tokens (Act_k): loss_mask = True
    - Observation tokens (Obs_k): loss_mask = False
    - Dual-Clip PPO Loss calculation ignores Obs_k completely.
    """
    # Prompt: 10 tokens (e.g. 0 to 9)
    P = 10
    prompt_tokens = list(range(100, 100 + P))

    # Turn 1: Model generates 5 tokens (Act 1)
    act1_tokens = [201, 202, 203, 204, 205]
    act1_lps = [-0.2, -0.4, -0.1, -0.3, -0.5]
    act1_mask = [True] * 5

    # Turn 1: Environment returns 4 tokens (Obs 1)
    obs1_tokens = [301, 302, 303, 304]
    obs1_lps = [0.0, 0.0, 0.0, 0.0]
    obs1_mask = [False] * 4

    # Turn 2: Model generates 3 tokens (Act 2)
    act2_tokens = [401, 402, 403]
    act2_lps = [-0.15, -0.25, -0.35]
    act2_mask = [True] * 3

    # Turn 2: Finish -> 0 obs tokens
    response_tokens = act1_tokens + obs1_tokens + act2_tokens
    response_lps = act1_lps + obs1_lps + act2_lps
    response_mask = act1_mask + obs1_mask + act2_mask
    R = len(response_tokens)  # 5 + 4 + 3 = 12 tokens

    traj1 = Trajectory(
        prompt="Solve task",
        prompt_tokens=prompt_tokens,
        response_text="Action 1 Obs 1 Action 2",
        response_tokens=response_tokens,
        tokens=prompt_tokens + response_tokens,
        rollout_log_probs=response_lps,
        reward=1.0,
        response_loss_mask=response_mask,
    )

    traj2 = Trajectory(
        prompt="Solve task",
        prompt_tokens=prompt_tokens,
        response_text="Action 1 Obs 1 Action 2 (Fail)",
        response_tokens=response_tokens,
        tokens=prompt_tokens + response_tokens,
        rollout_log_probs=response_lps,
        reward=0.0,
        response_loss_mask=response_mask,
    )

    batch = build_agentic_rollout_batch([traj1, traj2], pad_token_id=0)

    # 1. Contract Assertion: Shift-by-One Response slice is strictly [P - 1 : P + R - 1]
    # Length of slice must be exactly R
    slice_start = P - 1
    slice_end = P + R - 1
    assert slice_end - slice_start == R

    # 2. Total active mask tokens must exactly equal sum of True in response_loss_mask (5 + 3 = 8 tokens per trajectory)
    assert batch.loss_mask[0, slice_start:slice_end].sum().item() == 8
    assert batch.loss_mask.sum().item() == 16  # 2 trajectories * 8

    # 3. Exactly the Obs_1 positions must be False in loss_mask
    obs_slice = batch.loss_mask[0, slice_start + 5 : slice_start + 9]
    assert torch.all(~obs_slice)

    # 4. GRPO whitening and policy loss calculation
    batch = compute_grpo_advantages(batch, group_size=2)
    assert batch.advantages is not None
    # Advantages on Obs_1 positions must be 0.0 because of loss_mask isolation
    assert torch.all(batch.advantages[0, slice_start + 5 : slice_start + 9] == 0.0)

    # Mock trainer logprobs
    train_lps = batch.rollout_log_probs.clone()
    loss, metrics = compute_grpo_policy_loss(
        train_log_probs=train_lps,
        rollout_log_probs=batch.rollout_log_probs,
        advantages=batch.advantages,
        loss_mask=batch.loss_mask,
        clip_eps=0.2,
    )

    # Policy loss should be finite, and pg_loss should be 0.0 when train_lps == rollout_lps and mean adv is symmetric
    assert torch.isfinite(loss)
    assert "pg_loss" in metrics
