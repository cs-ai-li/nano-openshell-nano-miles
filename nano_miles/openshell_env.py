from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class SandboxConfig:
    """Configuration for OpenShell sandbox execution and timeouts."""
    openshell_dir: str = "nano-openshell"
    sock_path: str = "/tmp/nano-supervisor.sock"
    channel_token: str = "nano-mediated-token-v1"
    timeout_s: float = 10.0
    max_output_chars: int = 2000
    base_session_dir: str = "/tmp/nano_sessions"
    python_bin: str = sys.executable


@dataclass
class StepResult:
    """Step execution output following standard RL environment contract."""
    observation: str
    reward: float
    done: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class SupervisorContext:
    """
    Lightweight context manager to ensure host Supervisor is running.
    Spawns supervisor.py in background if /tmp/nano-supervisor.sock is not yet active.
    """

    def __init__(
        self,
        openshell_dir: str = "nano-openshell",
        sock_path: str = "/tmp/nano-supervisor.sock",
        channel_token: str = "nano-mediated-token-v1",
        real_openai_key: str = "sk-live-mock-key",
        python_bin: str = sys.executable,
    ) -> None:
        self.openshell_dir = Path(openshell_dir).resolve()
        self.sock_path = sock_path
        self.channel_token = channel_token
        self.real_openai_key = real_openai_key
        self.python_bin = python_bin
        self.proc: Optional[subprocess.Popen] = None
        self._we_spawned = False

    def is_alive(self) -> bool:
        if not os.path.exists(self.sock_path):
            return False
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect(self.sock_path)
            s.close()
            return True
        except (OSError, ConnectionRefusedError):
            return False

    def start(self) -> "SupervisorContext":
        if self.is_alive():
            return self

        # Clean dead socket if present
        if os.path.exists(self.sock_path):
            try:
                os.unlink(self.sock_path)
            except OSError:
                pass

        sup_script = self.openshell_dir / "supervisor.py"
        if not sup_script.is_file():
            raise FileNotFoundError(f"Supervisor script not found at: {sup_script}")

        env = os.environ.copy()
        env["NANO_SOCK_PATH"] = self.sock_path
        env["NANO_CHANNEL_TOKEN"] = self.channel_token
        env["REAL_OPENAI_KEY"] = self.real_openai_key

        self.proc = subprocess.Popen(
            [self.python_bin, str(sup_script)],
            cwd=str(self.openshell_dir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._we_spawned = True

        # Wait for socket to become active
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self.is_alive():
                return self
            time.sleep(0.05)

        self.stop()
        raise TimeoutError(f"Supervisor socket {self.sock_path} failed to initialize within 5.0s")

    def stop(self) -> None:
        if self._we_spawned and self.proc is not None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=2.0)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait(timeout=1.0)
                except Exception:
                    pass
            self.proc = None
            self._we_spawned = False

        if os.path.exists(self.sock_path):
            try:
                os.unlink(self.sock_path)
            except OSError:
                pass

    def __enter__(self) -> "SupervisorContext":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    async def __aenter__(self) -> "SupervisorContext":
        return self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def _kill_process_tree(pid: int) -> None:
    """Recursively kill a process, its child process groups, and all descendants."""
    import glob
    child_pids: List[int] = []
    for stat_file in glob.glob("/proc/[0-9]*/stat"):
        try:
            with open(stat_file, "r") as f:
                parts = f.read().split()
                if len(parts) >= 4 and int(parts[3]) == pid:
                    child_pids.append(int(parts[0]))
        except Exception:
            pass

    for cpid in child_pids:
        try:
            os.killpg(cpid, signal.SIGKILL)
        except Exception:
            try:
                os.kill(cpid, signal.SIGKILL)
            except Exception:
                pass

    try:
        os.killpg(pid, signal.SIGKILL)
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


class OpenShellSandbox:
    """
    Manages session directory isolation and asynchronous command execution
    inside nano-openshell Linux namespaces.
    """

    def __init__(self, session_id: str, config: Optional[SandboxConfig] = None) -> None:
        self.session_id = session_id
        self.config = config or SandboxConfig()
        self.cwd = Path(self.config.base_session_dir).resolve() / session_id
        self.cwd.mkdir(parents=True, exist_ok=True)
        self.sandbox_script = (Path(self.config.openshell_dir).resolve() / "sandbox.py")

    def format_output(self, stdout: str, stderr: str, exit_code: int) -> str:
        """Truncate long outputs and format into standard <tool_response> observation."""
        body = ""
        if stdout:
            body += stdout
        if stderr:
            if body and not body.endswith("\n"):
                body += "\n"
            body += f"[stderr]\n{stderr}"
        if not body:
            body = "(no output)"

        # Truncate if exceeds max_output_chars, keeping head and tail
        if len(body) > self.config.max_output_chars:
            half = (self.config.max_output_chars - 30) // 2
            body = body[:half] + "\n...[truncated]...\n" + body[-half:]

        return f"\n<tool_response>\n{body}\n[exit_code: {exit_code}]\n</tool_response>\n"

    async def execute_command(self, command: str) -> Tuple[int, str, str, bool]:
        """
        Execute command asynchronously inside the sandbox boundary.
        Uses native asyncio subprocess with a new process group for fail-safe cleanup.
        Returns: (exit_code, stdout, stderr, is_timeout)
        """
        if not self.sandbox_script.is_file():
            raise FileNotFoundError(f"sandbox.py not found at {self.sandbox_script}")

        env = os.environ.copy()
        env["NANO_SOCK_PATH"] = self.config.sock_path
        env["NANO_CHANNEL_TOKEN"] = self.config.channel_token

        args = [self.config.python_bin, str(self.sandbox_script), "--", "bash", "-c", command]

        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(self.cwd),
            env=env,
            start_new_session=True,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        is_timeout = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.timeout_s,
            )
            exit_code = proc.returncode if proc.returncode is not None else -1
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
        except (asyncio.TimeoutError, Exception) as exc:
            is_timeout = isinstance(exc, asyncio.TimeoutError)
            _kill_process_tree(proc.pid)
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except Exception:
                pass

            exit_code = 124 if is_timeout else -1
            stdout = ""
            stderr = f"Execution timeout after {self.config.timeout_s}s" if is_timeout else f"Subprocess error: {exc}"

        return exit_code, stdout, stderr, is_timeout

    def cleanup(self) -> None:
        """Safely destroy scratch session directory."""
        if self.cwd.exists():
            shutil.rmtree(self.cwd, ignore_errors=True)


def parse_action(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse model action from text.
    Returns: (action_type, payload)
    action_type can be 'bash', 'finish', or None.
    """
    bash_match = re.search(r"<bash>(.*?)</bash>", text, re.DOTALL)
    if bash_match:
        return "bash", bash_match.group(1).strip()

    finish_match = re.search(r"<finish>(.*?)</finish>", text, re.DOTALL)
    if finish_match:
        return "finish", finish_match.group(1).strip()

    return None, None


class OpenShellEnv:
    """
    Standard RL Environment wrapping OpenShellSandbox.
    Follows reset() -> prompt, step(action) -> StepResult contract.
    """

    def __init__(
        self,
        session_id: str,
        task_prompt: str,
        ground_truth: Any = None,
        config: Optional[SandboxConfig] = None,
        max_turns: int = 5,
        reward_fn: Optional[Callable[[str, Any, Dict[str, Any]], float]] = None,
    ) -> None:
        self.session_id = session_id
        self.task_prompt = task_prompt
        self.ground_truth = ground_truth
        self.config = config or SandboxConfig()
        self.max_turns = max_turns
        self.reward_fn = reward_fn
        self.sandbox = OpenShellSandbox(session_id, self.config)
        self.current_turn = 0
        self.is_done = False
        self.history: List[Dict[str, Any]] = []

    async def reset(self) -> str:
        """Reset environment, wipe scratch workspace, and return task prompt."""
        self.current_turn = 0
        self.is_done = False
        self.history.clear()
        self.sandbox.cleanup()
        self.sandbox.cwd.mkdir(parents=True, exist_ok=True)
        return self.task_prompt

    async def step(self, action_text: str) -> StepResult:
        """
        Execute one action turn.
        Parses <bash>...</bash> or <finish>...</finish>.
        """
        if self.is_done:
            raise RuntimeError(f"Step called on terminated environment {self.session_id}")

        self.current_turn += 1
        action_type, payload = parse_action(action_text)

        if action_type == "finish":
            self.is_done = True
            reward = 1.0
            if self.reward_fn:
                reward = self.reward_fn(payload or "", self.ground_truth, {"turn": self.current_turn})
            elif self.ground_truth is not None:
                reward = 1.0 if str(payload).strip() == str(self.ground_truth).strip() else 0.0

            obs = f"\n<task_finished>\nAnswer received: {payload}\n</task_finished>\n"
            return StepResult(
                observation=obs,
                reward=reward,
                done=True,
                metadata={"final_answer": payload, "turn": self.current_turn, "action_type": "finish"},
            )

        if action_type == "bash":
            cmd = payload or ""
            exit_code, stdout, stderr, is_timeout = await self.sandbox.execute_command(cmd)
            obs = self.sandbox.format_output(stdout, stderr, exit_code)

            step_reward = -0.02  # Minor step penalty to encourage shorter trajectories
            if is_timeout:
                step_reward -= 0.1
            elif exit_code != 0:
                step_reward -= 0.05
            else:
                step_reward += 0.1  # Dense shaping reward for valid execution with exit_code == 0

            reached_max = self.current_turn >= self.max_turns
            if reached_max:
                self.is_done = True

            meta = {
                "exit_code": exit_code,
                "is_timeout": is_timeout,
                "turn": self.current_turn,
                "action_type": "bash",
                "command": cmd,
            }
            return StepResult(observation=obs, reward=step_reward, done=self.is_done, metadata=meta)

        # Invalid format
        step_reward = -0.1
        reached_max = self.current_turn >= self.max_turns
        if reached_max:
            self.is_done = True

        obs = (
            "\n<tool_response>\n"
            "Format error: Please provide actions wrapped in <bash>command</bash> "
            "or complete the task with <finish>answer</finish>.\n"
            "[exit_code: 1]\n"
            "</tool_response>\n"
        )
        return StepResult(
            observation=obs,
            reward=step_reward,
            done=self.is_done,
            metadata={"turn": self.current_turn, "format_error": True, "action_type": None},
        )

    def close(self) -> None:
        """Clean up sandbox files."""
        self.sandbox.cleanup()
