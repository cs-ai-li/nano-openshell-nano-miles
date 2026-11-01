from __future__ import annotations

import asyncio
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import aiohttp
import torch
from nano_miles.openshell_env import OpenShellEnv, SandboxConfig
from nano_miles.rollout import SGLangClient, build_rollout_batch
from nano_miles.schema import RolloutBatch, Trajectory


class AgenticRolloutClient:
    """
    Manages multi-turn TITO rollouts between SGLang inference engine
    and OpenShellEnv sandboxes, assembling interleaved loss masks.
    """

    def __init__(
        self,
        sglang_client: SGLangClient,
        max_turns: int = 5,
        default_stop_sequences: Optional[List[str]] = None,
    ) -> None:
        self.client = sglang_client
        self.tokenizer = sglang_client.tokenizer
        self.max_turns = max_turns
        self.stop_sequences = default_stop_sequences or ["</bash>", "</finish>"]

    async def _generate_turn(
        self,
        prefix_tokens: List[int],
        sampling_params: Optional[Dict[str, Any]] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> Tuple[str, List[int], List[float], List[bool]]:
        """
        Generate one turn from SGLang given the accumulated prefix tokens.
        Enforces no_stop_trim: True so stop strings retain real tokens and logprobs.
        Any client-fallback padded tags are strictly marked with loss_mask = False.
        """
        params: Dict[str, Any] = {
            "temperature": 0.7,
            "max_new_tokens": 256,
            "top_p": 0.95,
            "stop": self.stop_sequences,
            "stop_token_ids": self.client.default_stop_token_ids,
            "no_stop_trim": True,
        }
        if sampling_params:
            params.update(sampling_params)

        payload = {
            "input_ids": prefix_tokens,
            "sampling_params": params,
            "return_logprob": True,
        }

        close_sess = session is None
        sess = session or aiohttp.ClientSession()
        try:
            async with sess.post(self.client.generate_url, json=payload) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"SGLang /generate failed {resp.status}: {await resp.text()}")
                data = await resp.json()
        finally:
            if close_sess:
                await sess.close()

        text = data.get("text", "")
        out_lps = data.get("meta_info", {}).get("output_token_logprobs", [])
        resp_tokens = [int(item[1]) for item in out_lps]
        resp_lps = [float(item[0]) for item in out_lps]
        # Real SGLang tokens get True mask
        resp_mask = [True] * len(resp_tokens)

        # Fallback: if model closed an opening tag without emitting closing tag before truncation
        for stop_tag in self.stop_sequences:
            open_tag = stop_tag.replace("/", "")
            if open_tag in text and stop_tag not in text:
                text += stop_tag
                tag_tokens = self.tokenizer.encode(stop_tag, add_special_tokens=False)
                resp_tokens.extend(tag_tokens)
                resp_lps.extend([0.0] * len(tag_tokens))
                # Client-synthesized tokens MUST be masked out (False)
                resp_mask.extend([False] * len(tag_tokens))

        return text, resp_tokens, resp_lps, resp_mask

    async def rollout_episode(
        self,
        env: OpenShellEnv,
        sampling_params: Optional[Dict[str, Any]] = None,
        session: Optional[aiohttp.ClientSession] = None,
        weight_version: int = 0,
    ) -> Trajectory:
        """
        Roll out a full multi-turn agentic trajectory in an OpenShell environment.
        Assembles interleaved response_loss_mask where model turns are True
        and environment observation turns are False.
        """
        initial_prompt = await env.reset()
        prompt_tokens = self.tokenizer.encode(initial_prompt, add_special_tokens=False)

        current_prefix_tokens = list(prompt_tokens)
        all_response_tokens: List[int] = []
        all_response_lps: List[float] = []
        all_response_mask: List[bool] = []
        full_response_text: List[str] = []

        total_reward = 0.0

        for turn in range(1, self.max_turns + 1):
            # 1. Model generation turn (Loss Mask from _generate_turn)
            model_text, gen_tokens, gen_lps, gen_mask = await self._generate_turn(
                current_prefix_tokens, sampling_params=sampling_params, session=session
            )

            all_response_tokens.extend(gen_tokens)
            all_response_lps.extend(gen_lps)
            all_response_mask.extend(gen_mask)
            full_response_text.append(model_text)
            current_prefix_tokens.extend(gen_tokens)

            # 2. Environment step execution
            step_result = await env.step(model_text)
            total_reward += step_result.reward

            if step_result.done:
                break

            # 3. Environment observation turn (Loss Mask = False, LogProbs = 0.0)
            obs_tokens = self.tokenizer.encode(step_result.observation, add_special_tokens=False)
            all_response_tokens.extend(obs_tokens)
            all_response_lps.extend([0.0] * len(obs_tokens))
            all_response_mask.extend([False] * len(obs_tokens))
            full_response_text.append(step_result.observation)
            current_prefix_tokens.extend(obs_tokens)

        complete_response_text = "".join(full_response_text)

        return Trajectory(
            prompt=initial_prompt,
            prompt_tokens=prompt_tokens,
            response_text=complete_response_text,
            response_tokens=all_response_tokens,
            tokens=prompt_tokens + all_response_tokens,
            rollout_log_probs=all_response_lps,
            reward=total_reward,
            weight_version=weight_version,
            response_loss_mask=all_response_mask,
        )

    async def rollout_group(
        self,
        envs: List[OpenShellEnv],
        sampling_params: Optional[Dict[str, Any]] = None,
        weight_version: int = 0,
    ) -> List[Trajectory]:
        """Roll out multiple environment sessions concurrently in parallel."""
        async with aiohttp.ClientSession() as sess:
            tasks = [
                self.rollout_episode(
                    env,
                    sampling_params=sampling_params,
                    session=sess,
                    weight_version=weight_version,
                )
                for env in envs
            ]
            return list(await asyncio.gather(*tasks))


def build_agentic_rollout_batch(
    trajectories: List[Trajectory], pad_token_id: int = 0
) -> RolloutBatch:
    """Collate multi-turn Trajectories into RolloutBatch with interleaved loss mask."""
    return build_rollout_batch(trajectories, pad_token_id=pad_token_id)
