import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_shape(path: Path):
    return tuple(int(x) for x in path.read_text().strip().split())


def load_bf16_hex(case_dir: Path, name: str):
    shape = load_shape(case_dir / f"{name}.shape.txt")
    values = []
    with open(case_dir / f"{name}.bf16_hex.txt") as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(int(line, 16))
    arr = np.asarray(values, dtype=np.uint16)
    return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16).reshape(shape)


def load_int(case_dir: Path, name: str):
    shape = load_shape(case_dir / f"{name}.shape.txt")
    return torch.from_numpy(np.loadtxt(case_dir / f"{name}.int.txt", dtype=np.int64)).reshape(shape)


def load_fp32(case_dir: Path, name: str):
    shape = load_shape(case_dir / f"{name}.shape.txt")
    return torch.from_numpy(np.loadtxt(case_dir / f"{name}.fp32.txt", dtype=np.float32)).reshape(shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir")
    args = parser.parse_args()
    case_dir = Path(args.case_dir)
    meta = json.loads((case_dir / "metadata.json").read_text())
    q = load_bf16_hex(case_dir, "q_after_query_up_rope")
    kv = load_bf16_hex(case_dir, "kv_after_rope_quant_compress")
    topk = load_int(case_dir, "topk_idxs")
    sink = load_fp32(case_dir, "attn_sink")
    expected = load_bf16_hex(case_dir, "qcau_o_after_inverse_rope")
    qcau_py = load_bf16_hex(case_dir, "qcau_py_o_after_inverse_rope")
    diff = (expected.float() - qcau_py.float()).abs()
    print("case:", meta["case"])
    print("q:", tuple(q.shape), q.dtype)
    print("kv:", tuple(kv.shape), kv.dtype)
    print("topk:", tuple(topk.shape), topk.dtype)
    print("sink:", tuple(sink.shape), sink.dtype)
    print("expected:", tuple(expected.shape), expected.dtype)
    print("max diff expected vs qcau_py:", diff.max().item())


if __name__ == "__main__":
    main()
