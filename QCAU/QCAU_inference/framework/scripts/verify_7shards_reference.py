import argparse
import gc
import importlib.util
import json
import os
import sys

import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_prefixed(module, ckpt_path, prefix):
    loaded = 0
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(prefix):
                set_module_tensor(module, key[len(prefix):], f.get_tensor(key))
                loaded += 1
    refresh_weight_scale_links(module)
    return [], [], loaded


def set_module_tensor(module, name, tensor):
    parts = name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    attr = parts[-1]
    if attr in parent._parameters:
        old = parent._parameters[attr]
        if (
            old is not None
            and hasattr(torch, "float4_e2m1fn_x2")
            and old.dtype == torch.float4_e2m1fn_x2
            and tensor.dtype == torch.int8
        ):
            tensor = tensor.view(torch.float4_e2m1fn_x2)
        elif old is not None and old.dtype != tensor.dtype:
            tensor = tensor.to(old.dtype)
        requires_grad = old.requires_grad if old is not None else False
        if not (tensor.is_floating_point() or tensor.is_complex()):
            requires_grad = False
        parent._parameters[attr] = nn.Parameter(tensor, requires_grad=requires_grad)
    elif attr in parent._buffers:
        parent._buffers[attr] = tensor
    else:
        setattr(parent, attr, tensor)


def refresh_weight_scale_links(module):
    for child in module.modules():
        if hasattr(child, "weight") and hasattr(child, "scale") and child.scale is not None:
            try:
                child.weight.scale = child.scale
            except Exception:
                pass


def dequant_fp8_weight_to_bf16(linear_module):
    if not hasattr(linear_module, "weight") or not hasattr(linear_module, "scale"):
        return
    if linear_module.weight.dtype != torch.float8_e4m3fn or linear_module.scale is None:
        return
    weight = linear_module.weight.detach()
    scale = linear_module.scale.detach().float()
    scale = scale.repeat_interleave(128, dim=0)[: weight.shape[0]]
    scale = scale.repeat_interleave(128, dim=1)[:, : weight.shape[1]]
    new_weight = (weight.float() * scale).to(torch.bfloat16)
    linear_module._parameters["weight"] = nn.Parameter(new_weight, requires_grad=False)
    linear_module._parameters["scale"] = None


def load_exact_embedding(embed, ckpt_path):
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        state = {"weight": f.get_tensor("embed.weight")}
    missing, unexpected = embed.load_state_dict(state, strict=False)
    return list(missing), list(unexpected)


def make_chunked_sparse_attn(orig_sparse_attn, head_chunk):
    def chunked(q, kv, attn_sink, topk_idxs, softmax_scale):
        if head_chunk <= 0 or head_chunk >= q.size(2):
            return orig_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)
        outs = []
        for start in range(0, q.size(2), head_chunk):
            end = min(start + head_chunk, q.size(2))
            outs.append(
                orig_sparse_attn(
                    q[:, :, start:end].contiguous(),
                    kv,
                    attn_sink[start:end].contiguous(),
                    topk_idxs,
                    softmax_scale,
                )
            )
        return torch.cat(outs, dim=2)
    return chunked


def free_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def cuda_mem(label):
    free, total = torch.cuda.mem_get_info()
    print(f"{label}: cuda free={free / 2**30:.2f}GiB total={total / 2**30:.2f}GiB")


def stream_block(model, args, layer_id, ckpt_path, h_cpu, input_ids_cpu, head_chunk):
    print(f"\n--- layer {layer_id} ---")
    cuda_mem("before construct")
    block = model.Block(layer_id, args)
    missing, unexpected, count = load_prefixed(block, ckpt_path, f"layers.{layer_id}.")
    dequant_fp8_weight_to_bf16(block.attn.wo_a)
    print(f"loaded tensors={count} missing={missing[:5]} unexpected={unexpected[:5]}")
    block.cuda().eval()
    cuda_mem("after load gpu")
    h = h_cpu.cuda()
    input_ids = input_ids_cpu.cuda()
    orig_sparse = model.sparse_attn
    model.sparse_attn = make_chunked_sparse_attn(orig_sparse, head_chunk)
    try:
        with torch.inference_mode():
            out = block(h, start_pos=0, input_ids=input_ids)
        out_cpu = out.detach().cpu()
    finally:
        model.sparse_attn = orig_sparse
        del block, h, input_ids
        free_cuda()
    cuda_mem("after free")
    return out_cpu


def sparse_attn_online_reference(q, kv, attn_sink, topk_idxs, softmax_scale, tile_size=64):
    bsz, seqlen, n_heads, head_dim = q.shape
    q_float = q.float()
    kv_float = kv.float()
    idxs = topk_idxs.to(device=q.device, dtype=torch.long)
    acc_o = torch.zeros(bsz, seqlen, n_heads, head_dim, device=q.device, dtype=torch.float32)
    sum_exp = torch.zeros(bsz, seqlen, n_heads, device=q.device, dtype=torch.float32)
    scores_max = torch.full((bsz, seqlen, n_heads), float("-inf"), device=q.device, dtype=torch.float32)
    for tile_base in range(0, idxs.size(-1), tile_size):
        tile_idxs = idxs[..., tile_base:tile_base + tile_size]
        valid = (tile_idxs >= 0) & (tile_idxs < kv.size(1))
        safe_idxs = tile_idxs.clamp(0, kv.size(1) - 1)
        gather_index = safe_idxs.unsqueeze(-1).expand(*safe_idxs.shape, head_dim)
        kv_tile = torch.gather(kv_float, 1, gather_index.reshape(bsz, -1, head_dim)).view(
            bsz, seqlen, -1, head_dim
        )
        kv_tile = torch.where(valid.unsqueeze(-1), kv_tile, torch.zeros_like(kv_tile))
        scores = torch.einsum("bshd,bstd->bsht", q_float, kv_tile) * softmax_scale
        scores = torch.where(valid.unsqueeze(2), scores, torch.full_like(scores, float("-inf")))
        previous_max = scores_max
        tile_has_valid = valid.any(dim=-1).unsqueeze(-1)
        tile_max = scores.amax(dim=-1)
        scores_max = torch.where(tile_has_valid, torch.maximum(previous_max, tile_max), previous_max)
        old_scale = torch.where(tile_has_valid, torch.exp(previous_max - scores_max), torch.ones_like(scores_max))
        safe_max = torch.where(torch.isfinite(scores_max), scores_max, torch.zeros_like(scores_max))
        probs = torch.exp(scores - safe_max.unsqueeze(-1))
        probs = torch.where(valid.unsqueeze(2), probs, torch.zeros_like(probs))
        probs_cast = probs.to(q.dtype).float()
        sum_exp = sum_exp * old_scale + probs.sum(dim=-1)
        acc_o = acc_o * old_scale.unsqueeze(-1) + torch.einsum("bsht,bstd->bshd", probs_cast, kv_tile)
    sink = attn_sink.float().view(1, 1, n_heads)
    sink_term = torch.exp(sink - scores_max)
    sum_exp = sum_exp + sink_term
    return (acc_o / sum_exp.clamp_min(torch.finfo(sum_exp.dtype).tiny).unsqueeze(-1)).to(q.dtype)


def verify_layer_qcau(orig_model, qcau_model, args, layer_id, layer_ckpt, h_cpu, head_chunk):
    print(f"\n--- layer {layer_id} QCAU verify ---")
    # Build the attention input exactly as Block.forward does before Attention.
    block = orig_model.Block(layer_id, args)
    missing, unexpected, count = load_prefixed(block, layer_ckpt, f"layers.{layer_id}.")
    print(f"block loaded tensors={count} missing={missing[:5]} unexpected={unexpected[:5]}")
    block.cuda().eval()
    h = h_cpu.cuda()
    with torch.inference_mode():
        x_attn, _, _ = block.hc_pre(h, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
        x_attn = block.attn_norm(x_attn)
    x_attn_cpu = x_attn.detach().cpu()
    del block, h, x_attn
    free_cuda()

    orig_attn = orig_model.Attention(layer_id, args)
    qcau_attn = qcau_model.Attention(layer_id, qcau_model.ModelArgs(**args.__dict__))
    load_prefixed(orig_attn, layer_ckpt, f"layers.{layer_id}.attn.")
    load_prefixed(qcau_attn, layer_ckpt, f"layers.{layer_id}.attn.")
    orig_attn.cuda().eval()
    qcau_attn.cuda().eval()
    x = x_attn_cpu.cuda()

    qcau_capture = {}
    real_qcau = qcau_model.QCAU

    def capture_qcau(**kwargs):
        out = real_qcau(**kwargs)
        qcau_capture.update(o=out, topk_idxs=kwargs["topk_idxs"], kv=kwargs["kv"], qr=kwargs["c_q"])
        raise RuntimeError("STOP_QCAU")

    qcau_model.QCAU = capture_qcau
    try:
        try:
            with torch.inference_mode():
                qcau_attn(x, start_pos=0)
        except RuntimeError as e:
            if str(e) != "STOP_QCAU":
                raise
    finally:
        qcau_model.QCAU = real_qcau

    orig_sparse = orig_model.sparse_attn
    orig_model.sparse_attn = make_chunked_sparse_attn(orig_sparse, head_chunk)
    orig_capture = {}

    def capture_sparse(q, kv, attn_sink, topk_idxs, softmax_scale):
        out = orig_model.sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)
        orig_capture.update(q=q, kv=kv, topk_idxs=topk_idxs, o=out)
        return out

    # Avoid recursion: use chunked sparse directly in local equivalent flow.
    try:
        with torch.inference_mode():
            bsz, seqlen, _ = x.size()
            freqs_cis = orig_attn.freqs_cis[:seqlen]
            rd = orig_attn.rope_head_dim
            if orig_attn.compressor.kv_cache is None:
                orig_attn.compressor.kv_cache = orig_attn.kv_cache[:, orig_attn.window_size:]
                orig_attn.compressor.freqs_cis = orig_attn.freqs_cis
                if orig_attn.indexer is not None:
                    orig_attn.indexer.freqs_cis = orig_attn.freqs_cis
            qr = orig_attn.q_norm(orig_attn.wq_a(x))
            q = orig_attn.wq_b(qr).unflatten(-1, (orig_attn.n_local_heads, orig_attn.head_dim))
            q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + orig_attn.eps)
            orig_model.apply_rotary_emb(q[..., -rd:], freqs_cis)
            kv = orig_attn.kv_norm(orig_attn.wkv(x))
            orig_model.apply_rotary_emb(kv[..., -rd:], freqs_cis)
            orig_model.act_quant(kv[..., :-rd], 64, orig_model.scale_fmt, orig_model.scale_dtype, True)
            topk_idxs = orig_model.get_window_topk_idxs(orig_attn.window_size, bsz, seqlen, 0).to(x.device)
            if orig_attn.indexer is not None:
                compress_topk_idxs = orig_attn.indexer(x, qr, 0, kv.size(1))
            else:
                compress_topk_idxs = orig_model.get_compress_topk_idxs(
                    orig_attn.compress_ratio,
                    bsz,
                    seqlen,
                    0,
                    kv.size(1),
                ).to(x.device)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1).int()
            orig_attn.kv_cache[:bsz, :seqlen] = kv
            kv_compress = orig_attn.compressor(x, 0)
            if kv_compress is not None:
                kv = torch.cat([kv, kv_compress], dim=1)
            o = make_chunked_sparse_attn(orig_sparse, head_chunk)(
                q, kv, orig_attn.attn_sink, topk_idxs, orig_attn.softmax_scale
            )
            orig_model.apply_rotary_emb(o[..., -rd:], freqs_cis, True)
            orig_capture.update(o=o, kv=kv, topk_idxs=topk_idxs, qr=qr)
    finally:
        orig_model.sparse_attn = orig_sparse

    diff = (qcau_capture["o"].float() - orig_capture["o"].float()).abs()
    metrics = {
        "topk_equal": torch.equal(qcau_capture["topk_idxs"], orig_capture["topk_idxs"]),
        "qr_max_abs_diff": (qcau_capture["qr"].float() - orig_capture["qr"].float()).abs().max().item(),
        "kv_max_abs_diff": (qcau_capture["kv"].float() - orig_capture["kv"].float()).abs().max().item(),
        "out_max_abs_diff": diff.max().item(),
        "out_mean_abs_diff": diff.mean().item(),
    }
    for key, value in metrics.items():
        print(f"{key}: {value}")
    tensors = {
        f"layer{layer_id}_x_attn": x_attn_cpu,
        f"layer{layer_id}_qcau_o": qcau_capture["o"].detach().cpu(),
        f"layer{layer_id}_orig_o": orig_capture["o"].detach().cpu(),
        f"layer{layer_id}_abs_diff": diff.detach().cpu(),
        f"layer{layer_id}_qcau_topk_idxs": qcau_capture["topk_idxs"].detach().cpu(),
        f"layer{layer_id}_orig_topk_idxs": orig_capture["topk_idxs"].detach().cpu(),
    }
    del orig_attn, qcau_attn, x
    free_cuda()
    return metrics, tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--head-chunk", type=int, default=16)
    parser.add_argument("--out", default="/root/danh/stream_available_shards_qcau_verify.safetensors")
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--max-layer", type=int, default=5)
    args_ns = parser.parse_args()

    inference_dir = "/root/danh/inference"
    qcau_dir = os.path.join(inference_dir, "QCAU")
    sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != qcau_dir]
    sys.modules.pop("QCAU", None)
    sys.path.insert(0, inference_dir)
    orig_model = load_module("stream_orig_model", os.path.join(inference_dir, "model.py"))
    import QCAU.model as qcau_model

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    with open("/root/danh/inference/config.json") as f:
        cfg = json.load(f)
    cfg["max_batch_size"] = 1
    cfg["max_seq_len"] = max(128, args_ns.seq_len)
    cfg["n_layers"] = args_ns.max_layer + 1
    margs = orig_model.ModelArgs(**cfg)
    qargs = qcau_model.ModelArgs(**cfg)
    # Initialize globals without building a full Transformer.
    for module, a in [(orig_model, margs), (qcau_model, qargs)]:
        module.world_size = 1
        module.rank = 0
        module.default_dtype = torch.float8_e4m3fn if a.dtype == "fp8" else torch.bfloat16
        module.scale_fmt = "ue8m0" if a.scale_dtype == "fp8" else a.scale_fmt
        module.scale_dtype = torch.float8_e8m0fnu if a.scale_dtype == "fp8" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained("/root/danh/weight")
    toks = tokenizer.encode(args_ns.prompt)[: args_ns.seq_len]
    toks = toks + [tokenizer.eos_token_id] * (args_ns.seq_len - len(toks))
    input_ids_cpu = torch.tensor([toks], dtype=torch.long, device="cpu")
    print("input_ids:", input_ids_cpu.tolist())

    embed = orig_model.ParallelEmbedding(margs.vocab_size, margs.dim)
    load_exact_embedding(embed, "/root/danh/weight/model-00001-of-00046.safetensors")
    embed.cuda().eval()
    with torch.inference_mode():
        h_cpu = embed(input_ids_cpu.cuda()).unsqueeze(2).repeat(1, 1, margs.hc_mult, 1).cpu()
    del embed
    free_cuda()
    print("after embed", tuple(h_cpu.shape), h_cpu.dtype)

    all_tensors = {}
    all_metrics = {}
    for layer_id in range(args_ns.max_layer + 1):
        ckpt = f"/root/danh/weight/model-{layer_id + 2:05d}-of-00046.safetensors"
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"missing shard for layer {layer_id}: {ckpt}")

        if layer_id >= 2:
            metrics, tensors = verify_layer_qcau(
                orig_model,
                qcau_model,
                margs,
                layer_id,
                ckpt,
                h_cpu,
                args_ns.head_chunk,
            )
            all_metrics[layer_id] = metrics
            all_tensors.update(tensors)

        if layer_id < args_ns.max_layer:
            h_cpu = stream_block(
                orig_model,
                margs,
                layer_id,
                ckpt,
                h_cpu,
                input_ids_cpu,
                args_ns.head_chunk,
            )

    save_file(all_tensors, args_ns.out)
    print("\n=== summary ===")
    for layer_id, metrics in all_metrics.items():
        print(
            f"layer {layer_id}: "
            f"topk_equal={metrics['topk_equal']} "
            f"qr_max_abs_diff={metrics['qr_max_abs_diff']} "
            f"kv_max_abs_diff={metrics['kv_max_abs_diff']} "
            f"out_max_abs_diff={metrics['out_max_abs_diff']} "
            f"out_mean_abs_diff={metrics['out_mean_abs_diff']}"
        )
    print("saved:", args_ns.out)


if __name__ == "__main__":
    main()
