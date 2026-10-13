# QCAU Framework

This folder is a portable package for verifying a rewritten QCAU implementation.
It contains reference model code plus standalone golden test vectors generated
from the real safetensor weights on the server. You do not need to bring the
safetensor weight shards to another machine if you only want to test QCAU.

## What To Download

Recommended:

```bash
scp -P 18680 root@47.224.206.51:/root/danh/framework_qcau_vectors.tar.gz .
tar -xzf framework_qcau_vectors.tar.gz
```

This archive contains everything under `/root/danh/framework`.

Alternative:

```bash
scp -P 18680 -r root@47.224.206.51:/root/danh/framework .
```

You do not need to download `/root/danh/weight/*.safetensors` for these offline
QCAU tests. Download the safetensors only if you want to regenerate new golden
vectors or rerun the full Python model.

## Folder Layout

```text
framework/
  README.md
  manifest.json
  config.json
  models/
    model_original.py
    kernel_original.py
    model_with_qcau.py
    QCAU.py
  scripts/
    load_vector_example.py
    verify_7shards_reference.py
  test_vectors/
    hello_seq1_layer2/
    hello_seq1_layer3/
    ...
```

`models/model_original.py` is the original model code.
`models/model_with_qcau.py` is the model variant that calls `QCAU(...)`.
`models/QCAU.py` is the current Python/TileLang QCAU implementation that matched
the original model bitwise on all generated cases.

`test_vectors/*` are the files you should use to test your C implementation.

## Generated Test Cases

There are 36 cases:

```text
3 prompts x 3 sequence lengths x 4 layers
prompts: hello, hardware, numbers
seq_len: 1, 4, 16
layers: 2, 3, 4, 5
```

All cases were generated from the real model weights and verified with:

```text
max_abs_diff(QCAU.py, original model) = 0.0
```

See `manifest.json` for the complete case list.

## Tensor File Format

Every tensor has a shape file:

```text
<name>.shape.txt
```

Values are stored flat in row-major order.

BF16 tensors are exact 16-bit hex dumps:

```text
<name>.bf16_hex.txt
```

Each line is one BF16 value as a 4-digit unsigned hex value. This is the exact
bit pattern, not a rounded decimal representation.

FP32 tensors use decimal text:

```text
<name>.fp32.txt
```

Integer tensors use decimal text:

```text
<name>.int.txt
```

Each case also has:

```text
metadata.json
```

This includes `softmax_scale`, `layer_id`, `seq_len`, `head_dim`,
`n_local_heads`, `rope_head_dim`, `compress_ratio`, and the diff recorded during
generation.

## Main Inputs For C QCAU Testing

If your C code implements the sparse-attention core only, use:

```text
q_after_query_up_rope.bf16_hex.txt
kv_after_rope_quant_compress.bf16_hex.txt
topk_idxs.int.txt
attn_sink.fp32.txt
softmax_scale from metadata.json
```

Expected sparse-attention output before inverse RoPE:

```text
sparse_o_before_inverse_rope.bf16_hex.txt
```

If your C code implements the whole QCAU output after inverse RoPE, compare
against:

```text
qcau_o_after_inverse_rope.bf16_hex.txt
```

Useful extra inputs:

```text
c_q.bf16_hex.txt
x_attn.bf16_hex.txt
freqs_cis_real.fp32.txt
freqs_cis_imag.fp32.txt
```

These help if you later move Query-Up or RoPE into C too.

## Quick Python Sanity Check

On a machine with Python, NumPy, and PyTorch installed:

```bash
python framework/scripts/load_vector_example.py \
  framework/test_vectors/hello_seq4_layer2
```

Expected output includes:

```text
max diff expected vs qcau_py: 0.0
```

This script is also a reference for how to read BF16 hex files.

## Recommended C Test Flow

1. Pick one small case first:

```text
framework/test_vectors/hello_seq1_layer2
```

2. Read shapes from:

```text
q_after_query_up_rope.shape.txt
kv_after_rope_quant_compress.shape.txt
topk_idxs.shape.txt
```

3. Read BF16 values as raw `uint16_t`, not decimal floats.

4. Run your C QCAU/sparse-attention implementation.

5. Dump your output as BF16 `uint16_t` hex in the same row-major order.

6. Compare bitwise against:

```text
sparse_o_before_inverse_rope.bf16_hex.txt
```

or, for full QCAU after inverse RoPE:

```text
qcau_o_after_inverse_rope.bf16_hex.txt
```

7. After the small case passes, run all cases in `manifest.json`.

## Notes

The original TileLang sparse attention was run with `head_chunk=16` on the RTX
4060 because launching all 64 heads at once exceeds the GPU shared-memory limit.
Chunking by head is mathematically exact because there is no cross-head
dependency.

For strict verification, compare BF16 bit patterns exactly. A small FP32 numeric
diff can still mean the BF16 golden output does not match bitwise.

## Verify All Saved Test Vectors

To check that the saved QCAU.py outputs match the original golden outputs for all 36 cases:

```bash
python framework/scripts/verify_all_vectors.py --framework framework
```

Expected final line:

```text
checked=36 failed=0
```

## Query-Up `w_uq` Files

The QCAU test vectors also include the exact `w_uq` data required by
`_query_up`:

```text
framework/w_uq.txt
framework/w_uq/layer2/
framework/w_uq/layer3/
framework/w_uq/layer4/
framework/w_uq/layer5/
```

The real model does not pass `w_uq` as a plain tensor. It passes:

```python
w_uq=self.wq_b
```

`self.wq_b` is a callable `Linear` module, so the actual branch is:

```python
if callable(w_uq):
    q_projected = w_uq(c_q)
```

For each layer, the exact files are:

```text
w_uq.weight.fp8_hex.txt  # FP8_E4M3, shape [32768, 1024]
w_uq.scale.e8m0_hex.txt  # FP8_E8M0, shape [256, 8]
*.shape.txt              # exact shapes
metadata.json            # dtype, layout, source shard, branch
```

The weight file has one output row per line. Each byte is represented by two
hex digits. The scale file has one scale row per line. These are raw bit
patterns, not decimal values.

For a vector case, read `layer_id` from its `metadata.json`, then use:

```text
../../w_uq/layer<layer_id>/w_uq.weight.fp8_hex.txt
../../w_uq/layer<layer_id>/w_uq.scale.e8m0_hex.txt
```

The `_query_up` sequence for the real model is:

```text
c_q BF16
  -> FP8 activation quantization, block size 128
  -> FP8 GEMM with w_uq.weight and w_uq.scale
  -> q_projected [B,S,32768]
  -> reshape [B,S,64,512]
  -> per-head RMS normalization
  -> forward RoPE
  -> q_after_query_up_rope
```

The generic `QCAU.py` also supports tensor forms of `w_uq`, but those branches
are not used by the real model vectors. They are documented in `w_uq.txt`:

```text
ndim == 2: [H*Dh, D_C], because _linear uses weight.transpose(-1,-2)
ndim == 3: [D_C, H, Dh]
ndim == 3: [H, D_C, Dh]
```

## `linear()` Act-Quant Variables for Callable `w_uq`

Because the real Query-Up path uses `w_uq=self.wq_b`, `_query_up` enters the
`callable(w_uq)` branch. Inside `wq_b.forward`, `model.linear(...)` runs:

```python
x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
return fp8_gemm(x, s, weight, weight.scale, scale_dtype)
```

For each test case, the framework now includes those two variables after
`act_quant`:

```text
linear_x_after_act_quant.fp8_e4m3_hex.txt
linear_x_after_act_quant.shape.txt
linear_s_after_act_quant.e8m0_hex.txt
linear_s_after_act_quant.shape.txt
linear_s_after_act_quant.preview_float.txt
```

Here:

```text
linear_x_after_act_quant: FP8_E4M3 raw uint8, shape [B, S, 1024]
linear_s_after_act_quant: FP8_E8M0 raw uint8, shape [B, S, 8]
```

`linear_s_after_act_quant` uses E8M0 exponent encoding:

```text
scale_value = 2 ** (uint8_value - 127)
```

These files correspond exactly to the local variables `x` and `s` in
`model.py::linear(...)` for the Query-Up `wq_b(c_q)` call.

