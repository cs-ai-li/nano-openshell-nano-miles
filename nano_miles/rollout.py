import asyncio
from typing import Any, Dict, List, Optional
import aiohttp
import torch
from transformers import AutoTokenizer
from nano_miles.schema import RolloutBatch, Trajectory


class SGLangClient:
    """Ultra-lightweight Token-in-Token-Out (TITO) client for SGLang Server."""

    def __init__(self, base_url: str = "http://127.0.0.1:30000", model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.base_url = base_url.rstrip("/")
        self.generate_url = f"{self.base_url}/generate"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.default_stop_token_ids = [self.tokenizer.eos_token_id, 151645, 151643]

    async def generate(
        self, prompt: str, sampling_params: Optional[Dict[str, Any]] = None, session: Optional[aiohttp.ClientSession] = None
    ) -> Trajectory:
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        params = {"temperature": 0.7, "max_new_tokens": 128, "top_p": 0.95, "stop_token_ids": self.default_stop_token_ids}
        if sampling_params:
            params.update(sampling_params)
            params.setdefault("stop_token_ids", self.default_stop_token_ids)

        payload = {"input_ids": prompt_tokens, "sampling_params": params, "return_logprob": True}
        close_sess = session is None
        sess = session or aiohttp.ClientSession()
        try:
            async with sess.post(self.generate_url, json=payload) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"SGLang /generate failed {resp.status}: {await resp.text()}")
                data = await resp.json()
        finally:
            if close_sess:
                await sess.close()

        out_lps = data.get("meta_info", {}).get("output_token_logprobs", [])
        resp_tokens = [int(item[1]) for item in out_lps]
        resp_lps = [float(item[0]) for item in out_lps]
        return Trajectory(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            response_text=data.get("text", ""),
            response_tokens=resp_tokens,
            tokens=prompt_tokens + resp_tokens,
            rollout_log_probs=resp_lps,
        )

    def generate_sync(self, prompt: str, sampling_params: Optional[Dict[str, Any]] = None) -> Trajectory:
        return asyncio.run(self.generate(prompt, sampling_params))

    async def generate_group(
        self, prompts: List[str], group_size: int, sampling_params: Optional[Dict[str, Any]] = None
    ) -> List[Trajectory]:
        async with aiohttp.ClientSession() as sess:
            tasks = [self.generate(p, sampling_params, session=sess) for p in prompts for _ in range(group_size)]
            return list(await asyncio.gather(*tasks))

    def generate_group_sync(
        self, prompts: List[str], group_size: int, sampling_params: Optional[Dict[str, Any]] = None
    ) -> List[Trajectory]:
        return asyncio.run(self.generate_group(prompts, group_size, sampling_params))

    async def score_prefill_logprobs(
        self, tokens: List[int], prompt_len: int, session: Optional[aiohttp.ClientSession] = None
    ) -> List[float]:
        payload = {
            "input_ids": tokens,
            "sampling_params": {"max_new_tokens": 0, "temperature": 0.0, "skip_special_tokens": False},
            "return_logprob": True,
            "logprob_start_len": prompt_len - 1,
        }
        close_sess = session is None
        sess = session or aiohttp.ClientSession()
        try:
            async with sess.post(self.generate_url, json=payload) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"SGLang prefill failed {resp.status}: {await resp.text()}")
                data = await resp.json()
        finally:
            if close_sess:
                await sess.close()

        input_lps = data.get("meta_info", {}).get("input_token_logprobs", [])
        res = [float(item[0]) for item in input_lps[1:]]
        assert len(res) == len(tokens) - prompt_len
        return res

    def score_prefill_logprobs_sync(self, tokens: List[int], prompt_len: int) -> List[float]:
        return asyncio.run(self.score_prefill_logprobs(tokens, prompt_len))


def build_rollout_batch(trajectories: List[Trajectory], pad_token_id: int = 0) -> RolloutBatch:
    """Collate Trajectories into RolloutBatch with right-padding and Shift-by-One alignment."""
    if not trajectories:
        raise ValueError("Empty trajectories list")
    B = len(trajectories)
    max_len = max(t.total_len for t in trajectories)
    tokens = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    loss_mask = torch.zeros((B, max_len - 1), dtype=torch.bool)
    rollout_lps = torch.zeros((B, max_len - 1), dtype=torch.float32)
    rewards = torch.tensor([t.reward for t in trajectories], dtype=torch.float32)
    has_adv = any(t.advantages is not None for t in trajectories)
    advantages = torch.zeros((B, max_len - 1), dtype=torch.float32) if has_adv else None

    for i, t in enumerate(trajectories):
        P, R, total_len = t.prompt_len, t.response_len, t.total_len
        tokens[i, :total_len] = torch.tensor(t.tokens, dtype=torch.long)
        if t.response_loss_mask is not None:
            loss_mask[i, P - 1 : P + R - 1] = torch.tensor(t.response_loss_mask, dtype=torch.bool)
        else:
            loss_mask[i, P - 1 : P + R - 1] = True
        rollout_lps[i, P - 1 : P + R - 1] = torch.tensor(t.rollout_log_probs, dtype=torch.float32)
        if advantages is not None and t.advantages is not None:
            advantages[i, P - 1 : P + R - 1] = torch.tensor(t.advantages, dtype=torch.float32)

    expected_valid_tokens = sum(
        sum(t.response_loss_mask) if t.response_loss_mask is not None else t.response_len
        for t in trajectories
    )
    assert loss_mask.sum().item() == expected_valid_tokens
    return RolloutBatch(
        tokens=tokens, loss_mask=loss_mask, rollout_log_probs=rollout_lps,
        advantages=advantages, rewards=rewards, trajectories=trajectories,
        weight_version=max((t.weight_version for t in trajectories), default=0),
    )
