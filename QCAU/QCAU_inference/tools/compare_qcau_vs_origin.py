#!/usr/bin/env python3
"""Compare QCAU Attention.forward against the original implementation.

This is a lightweight correctness test. It instantiates tiny Attention modules,
copies identical random weights into both implementations, runs prefill and
decode, and reports bitwise/allclose differences.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_origin_model():
    """Load Deepseek_origin/inference/model.py with the PyTorch fallback kernel."""
    from dinhanh.QCAU.inference import kernel_fallback

    origin_dir = ROOT / "Deepseek_origin" / "inference"
    sys.path.insert(0, str(origin_dir))
    sys.modules["kernel"] = kernel_fallback
    spec = importlib.util.spec_from_file_location(
        "deepseek_origin_model_under_test",
        origin_dir / "model.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load Deepseek_origin/inference/model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_args(model_module, compress_ratio: int):
    return model_module.ModelArgs(
        max_batch_size=2,
        max_seq_len=32,
        dtype="bf16",
        scale_fmt=None,
        scale_dtype="fp32",
        expert_dtype=None,
        vocab_size=128,
        dim=128,
        moe_inter_dim=64,
        n_layers=1,
        n_hash_layers=0,
        n_mtp_layers=0,
        n_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        q_lora_rank=64,
        head_dim=128,
        rope_head_dim=64,
        o_groups=1,
        o_lora_rank=64,
        window_size=8,
        compress_ratios=(compress_ratio,),
        index_n_heads=4,
        index_head_dim=64,
        index_topk=4,
        hc_mult=2,
    )


def args_from_inference_config(model_module, config_path: Path):
    with config_path.open() as f:
        return model_module.ModelArgs(**json.load(f))


def truncated_args_from_config(model_module, config_path: Path, n_layers: int):
    with config_path.open() as f:
        data = json.load(f)
    data["n_layers"] = n_layers
    data["n_mtp_layers"] = 0
    data["compress_ratios"] = data["compress_ratios"][:n_layers]
    return model_module.ModelArgs(**data)


def reset_runtime_state():
    from dinhanh.QCAU.inference import model as qcau_model

    qcau_model.precompute_freqs_cis.cache_clear()
    qcau_model.get_window_topk_idxs.cache_clear()
    qcau_model.get_compress_topk_idxs.cache_clear()


def cpu_hadamard_rotation(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16:
        raise AssertionError("Hadamard rotation requires bf16 in the source model")
    n = x.size(-1)
    if n <= 0 or n & (n - 1):
        raise ValueError("Hadamard rotation requires a power-of-two width")
    original_shape = x.shape
    y = x.float().reshape(-1, n)
    span = 1
    while span < n:
        y = y.reshape(-1, n // (2 * span), 2 * span)
        left = y[..., :span].clone()
        right = y[..., span:].clone()
        y = torch.cat((left + right, left - right), dim=-1).reshape(-1, n)
        span *= 2
    return (y.reshape(original_shape) * n**-0.5).to(x.dtype)


def compare_tensors(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> bool:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    max_abs = diff.max().item()
    max_rel = (diff / expected_f.abs().clamp_min(1e-8)).max().item()
    bitwise = torch.equal(actual, expected)
    close = torch.allclose(actual_f, expected_f, atol=atol, rtol=rtol)
    print(
        f"{name}: bitwise={bitwise} allclose={close} "
        f"max_abs={max_abs:.8g} max_rel={max_rel:.8g}"
    )
    return close


@torch.inference_mode()
def run_case(compress_ratio: int, device: torch.device, dtype: torch.dtype, atol: float, rtol: float) -> bool:
    origin_model = load_origin_model()
    from dinhanh.QCAU.inference import model as qcau_model
    if device.type == "cpu":
        origin_model.rotate_activation = cpu_hadamard_rotation
        qcau_model.rotate_activation = cpu_hadamard_rotation

    torch.manual_seed(20260915 + compress_ratio)
    torch.set_default_dtype(dtype)
    reset_runtime_state()

    args_origin = tiny_args(origin_model, compress_ratio)
    args_qcau = tiny_args(qcau_model, compress_ratio)
    origin_model.default_dtype = dtype
    qcau_model.default_dtype = dtype
    origin_model.scale_fmt = None
    qcau_model.scale_fmt = None
    origin_model.scale_dtype = torch.float32
    qcau_model.scale_dtype = torch.float32
    origin = origin_model.Attention(0, args_origin).to(device)
    qcau = qcau_model.Attention(0, args_qcau).to(device)
    for param in origin.parameters():
        if param.is_floating_point():
            param.normal_(mean=0.0, std=0.02)
    qcau.load_state_dict(origin.state_dict(), strict=True)
    if dtype == torch.float32:
        origin.float()
        qcau.float()
    origin.eval()
    qcau.eval()

    prefill_x = torch.randn(2, 12, args_origin.dim, device=device, dtype=dtype)
    origin_prefill = origin(prefill_x, 0)
    qcau_prefill = qcau(prefill_x, 0)
    ok = compare_tensors(
        f"compress_ratio={compress_ratio} prefill",
        qcau_prefill,
        origin_prefill,
        atol,
        rtol,
    )

    decode_x = torch.randn(2, 1, args_origin.dim, device=device, dtype=dtype)
    origin_decode = origin(decode_x, prefill_x.size(1))
    qcau_decode = qcau(decode_x, prefill_x.size(1))
    ok &= compare_tensors(
        f"compress_ratio={compress_ratio} decode",
        qcau_decode,
        origin_decode,
        atol,
        rtol,
    )
    return ok


def shard_for_layer(index_path: Path, layer_id: int) -> Path:
    with index_path.open() as f:
        weight_map = json.load(f)["weight_map"]
    key = f"layers.{layer_id}.attn.wq_a.weight"
    shard = weight_map.get(key)
    if shard is None:
        raise KeyError(f"{key} not found in {index_path}")
    return index_path.parent / shard


def load_attention_state(shard_path: Path, layer_id: int):
    raw = load_file(str(shard_path), device="cpu")
    prefix = f"layers.{layer_id}.attn."
    state = {}
    ignored = []
    for key, value in raw.items():
        if key.startswith(prefix):
            stripped = key.removeprefix(prefix)
            if stripped.endswith(".scale") and stripped.startswith("wo_a."):
                ignored.append(stripped)
                continue
            state[stripped] = value
    if not state:
        raise RuntimeError(f"No attention tensors for layer {layer_id} in {shard_path}")
    return state, ignored


def first_n_layer_ids(ckpt_path: Path, n_layers: int) -> list[int]:
    index_path = ckpt_path / "model.safetensors.index.json"
    with index_path.open() as f:
        weight_map = json.load(f)["weight_map"]
    available = sorted(
        {
            int(key.split(".")[1])
            for key, shard in weight_map.items()
            if key.startswith("layers.") and (ckpt_path / shard).exists()
        }
    )
    wanted = list(range(n_layers))
    missing = [layer_id for layer_id in wanted if layer_id not in available]
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoint shards for layers {missing}; available layers are {available}"
        )
    return wanted


def load_partial_transformer_state(ckpt_path: Path, layer_ids: list[int]):
    index_path = ckpt_path / "model.safetensors.index.json"
    with index_path.open() as f:
        weight_map = json.load(f)["weight_map"]
    needed_prefixes = ["embed."] + [f"layers.{layer_id}." for layer_id in layer_ids]
    needed_shards = sorted(
        {
            shard
            for key, shard in weight_map.items()
            if any(key.startswith(prefix) for prefix in needed_prefixes)
        }
    )
    state = {}
    for shard in needed_shards:
        shard_path = ckpt_path / shard
        if not shard_path.exists():
            raise FileNotFoundError(f"Needed shard is missing: {shard_path}")
        raw = load_file(str(shard_path), device="cpu")
        for key, value in raw.items():
            if any(key.startswith(prefix) for prefix in needed_prefixes):
                if key.endswith(".scale") and ".attn.wo_a." in key:
                    continue
                state[key] = value
    return state, needed_shards


def validate_partial_load(load_result, allowed_missing_prefixes: tuple[str, ...]):
    missing, unexpected = load_result
    bad_missing = [
        key for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"Partial transformer load mismatch: missing={bad_missing} unexpected={unexpected}"
        )


def replace_parameter(module: torch.nn.Module, name: str, value: torch.Tensor):
    parts = name.split(".")
    parent = module.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else module
    leaf = parts[-1]
    if leaf not in parent._parameters:
        raise KeyError(f"{name} is not a parameter")
    parent._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
    if leaf == "weight" and hasattr(parent, "scale"):
        parent.weight.scale = parent.scale


def load_state_allowing_fp4(module: torch.nn.Module, state: dict[str, torch.Tensor], allowed_missing_prefixes: tuple[str, ...]):
    own_state = module.state_dict()
    unexpected = sorted(set(state) - set(own_state))
    missing = sorted(set(own_state) - set(state))
    bad_missing = [
        key for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if unexpected or bad_missing:
        raise RuntimeError(
            f"Partial transformer load mismatch: missing={bad_missing} unexpected={unexpected}"
        )
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    for key, value in state.items():
        target = own_state[key]
        value = value.to(device=target.device)
        if fp4_dtype is not None and target.dtype == fp4_dtype and value.dtype == torch.int8:
            value = value.view(fp4_dtype)
        if fp4_dtype is not None and (value.dtype == fp4_dtype or target.dtype == fp4_dtype):
            replace_parameter(module, key, value)
        else:
            target.copy_(value)


@torch.inference_mode()
def run_truncated_transformer(
    ckpt_path: Path,
    config_path: Path,
    n_layers: int,
    device: torch.device,
    prefill_len: int,
    atol: float,
    rtol: float,
) -> bool:
    origin_model = load_origin_model()
    from dinhanh.QCAU.inference import model as qcau_model

    if device.type == "cpu":
        origin_model.rotate_activation = cpu_hadamard_rotation
        qcau_model.rotate_activation = cpu_hadamard_rotation

    torch.manual_seed(12345)
    torch.set_default_dtype(torch.bfloat16)
    reset_runtime_state()
    for module in (origin_model, qcau_model):
        module.default_dtype = torch.float8_e4m3fn
        module.scale_fmt = "ue8m0"
        module.scale_dtype = getattr(torch, "float8_e8m0fnu", torch.float32)

    layer_ids = first_n_layer_ids(ckpt_path, n_layers)
    args_origin = truncated_args_from_config(origin_model, config_path, n_layers)
    args_qcau = truncated_args_from_config(qcau_model, config_path, n_layers)
    origin = origin_model.Transformer(args_origin).to(device)
    qcau = qcau_model.Transformer(args_qcau).to(device)
    state, shards = load_partial_transformer_state(ckpt_path, layer_ids)
    allowed_missing = ("norm.", "head.", "hc_head_", "mtp.")
    load_state_allowing_fp4(origin, state, allowed_missing)
    load_state_allowing_fp4(qcau, state, allowed_missing)
    origin.eval()
    qcau.eval()
    print(f"loaded shards: {', '.join(shards)}")

    input_ids = torch.randint(
        0,
        args_origin.vocab_size,
        (1, prefill_len),
        device=device,
        dtype=torch.long,
    )

    def forward_hidden(model, tokens, start_pos):
        h = model.embed(tokens)
        h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        for layer in model.layers:
            h = layer(h, start_pos, tokens)
        return h

    origin_prefill = forward_hidden(origin, input_ids, 0)
    qcau_prefill = forward_hidden(qcau, input_ids, 0)
    ok = compare_tensors(
        f"truncated transformer layers=0..{n_layers - 1} prefill_hidden",
        qcau_prefill,
        origin_prefill,
        atol,
        rtol,
    )

    decode_ids = torch.randint(
        0,
        args_origin.vocab_size,
        (1, 1),
        device=device,
        dtype=torch.long,
    )
    origin_decode = forward_hidden(origin, decode_ids, prefill_len)
    qcau_decode = forward_hidden(qcau, decode_ids, prefill_len)
    ok &= compare_tensors(
        f"truncated transformer layers=0..{n_layers - 1} decode_hidden",
        qcau_decode,
        origin_decode,
        atol,
        rtol,
    )
    return ok


@torch.inference_mode()
def run_checkpoint_layer(
    layer_id: int,
    ckpt_path: Path,
    config_path: Path,
    device: torch.device,
    prefill_len: int,
    atol: float,
    rtol: float,
) -> bool:
    origin_model = load_origin_model()
    from dinhanh.QCAU.inference import model as qcau_model

    if device.type == "cpu":
        origin_model.rotate_activation = cpu_hadamard_rotation
        qcau_model.rotate_activation = cpu_hadamard_rotation

    torch.manual_seed(9090 + layer_id)
    torch.set_default_dtype(torch.bfloat16)
    reset_runtime_state()
    for module in (origin_model, qcau_model):
        module.default_dtype = torch.float8_e4m3fn
        module.scale_fmt = "ue8m0"
        module.scale_dtype = getattr(torch, "float8_e8m0fnu", torch.float32)

    args_origin = args_from_inference_config(origin_model, config_path)
    args_qcau = args_from_inference_config(qcau_model, config_path)
    origin = origin_model.Attention(layer_id, args_origin).to(device)
    qcau = qcau_model.Attention(layer_id, args_qcau).to(device)

    shard = shard_for_layer(ckpt_path / "model.safetensors.index.json", layer_id)
    if not shard.exists():
        raise FileNotFoundError(f"Needed shard for layer {layer_id} is missing: {shard}")
    state, ignored = load_attention_state(shard, layer_id)
    missing_o, unexpected_o = origin.load_state_dict(state, strict=False)
    missing_q, unexpected_q = qcau.load_state_dict(state, strict=False)
    allowed_missing = set()
    bad_missing = (set(missing_o) | set(missing_q)) - allowed_missing
    bad_unexpected = (set(unexpected_o) | set(unexpected_q)) - set(ignored)
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Layer {layer_id} load mismatch: missing={sorted(bad_missing)} "
            f"unexpected={sorted(bad_unexpected)} ignored={ignored}"
        )
    origin.eval()
    qcau.eval()

    x = torch.randn(
        1,
        prefill_len,
        args_origin.dim,
        device=device,
        dtype=torch.bfloat16,
    )
    origin_prefill = origin(x, 0)
    qcau_prefill = qcau(x, 0)
    ok = compare_tensors(
        f"checkpoint layer={layer_id} ratio={args_origin.compress_ratios[layer_id]} prefill",
        qcau_prefill,
        origin_prefill,
        atol,
        rtol,
    )

    decode_x = torch.randn(1, 1, args_origin.dim, device=device, dtype=torch.bfloat16)
    origin_decode = origin(decode_x, prefill_len)
    qcau_decode = qcau(decode_x, prefill_len)
    ok &= compare_tensors(
        f"checkpoint layer={layer_id} ratio={args_origin.compress_ratios[layer_id]} decode",
        qcau_decode,
        origin_decode,
        atol,
        rtol,
    )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "QCAU" / "inference" / "config.json")
    parser.add_argument("--end-to-end", action="store_true")
    parser.add_argument("--n-layers", type=int, default=5)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--prefill-len", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--compress-ratios", type=int, nargs="+", default=[0, 4])
    args = parser.parse_args()

    device = torch.device(args.device)
    all_ok = True
    if args.checkpoint_path is not None and args.end_to_end:
        all_ok &= run_truncated_transformer(
            args.checkpoint_path,
            args.config,
            args.n_layers,
            device,
            args.prefill_len,
            args.atol,
            args.rtol,
        )
    elif args.checkpoint_path is not None:
        layers = args.layers if args.layers is not None else [0, 1, 2, 3, 4]
        for layer_id in layers:
            all_ok &= run_checkpoint_layer(
                layer_id,
                args.checkpoint_path,
                args.config,
                device,
                args.prefill_len,
                args.atol,
                args.rtol,
            )
    else:
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        for ratio in args.compress_ratios:
            all_ok &= run_case(ratio, device, dtype, args.atol, args.rtol)
    if all_ok:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
