import time
from typing import Any, Dict, Optional, Union
import requests
import torch
import torch.nn as nn
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket


def sync_weights_to_sglang(
    model: nn.Module,
    sglang_url: str = "http://127.0.0.1:30000",
    target_device: Optional[Union[str, torch.device]] = None,
    flush_cache: bool = True,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    Zero-disk CUDA IPC weight hot-update protocol.
    Streams weights from PyTorch model to SGLang server in memory via CUDA IPC.
    Supports cross-GPU staging when Trainer and SGLang are on different GPUs.

    Safety contracts:
      1. Open session with begin_weight_update.
      2. If target_device != model_device: stage tensors onto target_device via DMA.
      3. Synchronize both model_device and target_device streams.
      4. Tied weights: explicitly include lm_head.weight if tied.
      5. Keep bucket references alive until update_weights_from_tensor returns 200 OK.
      6. flush_cache=True to invalidate RadixCache.
      7. Close session with end_weight_update and clean up staging memory.
    """
    t0 = time.time()
    session = requests.Session()

    # Determine source and target devices
    model_param = next(model.parameters(), None)
    model_device = model_param.device if model_param is not None else torch.device("cuda:0")
    if target_device is not None:
        target_device = torch.device(target_device)
    else:
        target_device = model_device

    # 1. Open weight-update session (with self-healing for leaked sessions)
    resp = session.post(
        f"{sglang_url}/begin_weight_update",
        json={"selector": "all", "sync_base": True},
        timeout=timeout,
    )
    if resp.status_code == 400 and "already open" in resp.text:
        # Force clear previous leaked session lock and re-open
        session.post(f"{sglang_url}/end_weight_update", json={}, timeout=timeout)
        resp = session.post(
            f"{sglang_url}/begin_weight_update",
            json={"selector": "all", "sync_base": True},
            timeout=timeout,
        )

    if resp.status_code != 200 or not resp.json().get("success", False):
        raise RuntimeError(f"begin_weight_update failed: HTTP {resp.status_code} {resp.text}")

    # 2. Collect model weights on source device & handle tied weights
    named_tensors = [(name, p.detach().contiguous()) for name, p in model.named_parameters()]
    param_names = {n for n, _ in named_tensors}
    if hasattr(model, "lm_head") and "lm_head.weight" not in param_names:
        named_tensors.append(("lm_head.weight", model.lm_head.weight.detach().contiguous()))

    # 3. Create FlattenedTensorBucket on source device
    bucket = FlattenedTensorBucket(named_tensors=named_tensors)
    flat_tensor = bucket.get_flattened_tensor()

    # 4. If cross-device, perform a single contiguous PCIe DMA transfer to target device
    is_cross_device = target_device != model_device
    if is_cross_device:
        flat_tensor = flat_tensor.to(target_device, non_blocking=True)
        torch.cuda.synchronize(model_device)
        torch.cuda.synchronize(target_device)
    else:
        torch.cuda.synchronize(model_device)

    bucket_data = {
        "flattened_tensor": flat_tensor,
        "metadata": bucket.get_metadata(),
    }
    serialized_str = MultiprocessingSerializer.serialize(bucket_data, output_str=True)

    # 5. Dispatch IPC handle via HTTP while keeping bucket_data alive in GPU memory
    try:
        resp = session.post(
            f"{sglang_url}/update_weights_from_tensor",
            json={
                "serialized_named_tensors": [serialized_str],
                "load_format": "flattened_bucket",
                "flush_cache": flush_cache,
            },
            timeout=timeout,
        )
        if resp.status_code != 200 or not resp.json().get("success", False):
            raise RuntimeError(f"update_weights_from_tensor failed: HTTP {resp.status_code} {resp.text}")
    finally:
        # 6. Close session before releasing tensor references
        session.post(f"{sglang_url}/end_weight_update", json={}, timeout=timeout)
        del flat_tensor
        del bucket_data
        del bucket
        del named_tensors
        if is_cross_device:
            torch.cuda.synchronize(target_device)

    if is_cross_device:
        torch.cuda.synchronize(target_device)
    else:
        torch.cuda.synchronize(model_device)

    latency = time.time() - t0
    return {"success": True, "latency_seconds": latency, "target_device": str(target_device)}
