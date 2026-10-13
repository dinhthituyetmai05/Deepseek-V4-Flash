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


def load_fp32(case_dir: Path, name: str):
    shape = load_shape(case_dir / f"{name}.shape.txt")
    data = np.loadtxt(case_dir / f"{name}.fp32.txt", dtype=np.float32)
    return torch.from_numpy(np.asarray(data, dtype=np.float32)).reshape(shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", default="/root/danh/framework")
    args = parser.parse_args()

    framework = Path(args.framework)
    manifest = json.loads((framework / "manifest.json").read_text())
    failed = []
    for case in manifest["cases"]:
        case_dir = framework / "test_vectors" / case
        expected = load_bf16_hex(case_dir, "qcau_o_after_inverse_rope")
        qcau_py = load_bf16_hex(case_dir, "qcau_py_o_after_inverse_rope")
        abs_diff = load_fp32(case_dir, "abs_diff_qcau_py_vs_original")
        bitwise = torch.equal(expected, qcau_py)
        max_diff = (expected.float() - qcau_py.float()).abs().max().item()
        saved_max_diff = abs_diff.max().item()
        ok = bitwise and max_diff == 0.0 and saved_max_diff == 0.0
        print(
            f"{case}: bitwise={bitwise} max_diff={max_diff} "
            f"saved_abs_diff_max={saved_max_diff}"
        )
        if not ok:
            failed.append((case, bitwise, max_diff, saved_max_diff))

    print(f"\nchecked={len(manifest['cases'])} failed={len(failed)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
