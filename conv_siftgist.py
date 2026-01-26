# vecs_to_text.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def fvecs_memmap(path: Path):
    x = np.memmap(path, dtype=np.int32, mode="r")
    if x.size == 0:
        raise ValueError("Empty file")
    d = int(x[0])
    if x.size % (d + 1) != 0:
        raise ValueError(f"Invalid fvecs (size not divisible by d+1). d={d}, size={x.size}")
    n = x.size // (d + 1)
    x = x.reshape(n, d + 1)
    # data part is stored as float32 bytes; reinterpret
    return x, n, d

def bvecs_memmap(path: Path):
    d = int(np.fromfile(path, dtype=np.int32, count=1)[0])
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    rec = d + 4  # 4 bytes for int32 dimension header
    if mm.size % rec != 0:
        raise ValueError(f"Invalid bvecs (size not divisible by d+4). d={d}, size={mm.size}")
    n = mm.size // rec
    mm = mm.reshape(n, rec)
    return mm, n, d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input file (.fvecs or .bvecs)")
    ap.add_argument("-o", "--output", required=True, help="output text file (.txt)")
    ap.add_argument("--chunk", type=int, default=10000, help="rows per write (default: 10000)")
    ap.add_argument("--head", type=int, default=None, help="convert only first N vectors (recommended for huge datasets)")
    ap.add_argument("--fmt", default=None, help="number format, e.g. '%.6g' for floats or '%d' for ints")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    suf = in_path.suffix.lower()

    if suf == ".fvecs":
        mm, n, d = fvecs_memmap(in_path)
        kind = "fvecs"
        default_fmt = "%.6g"
    elif suf == ".bvecs":
        mm, n, d = bvecs_memmap(in_path)
        kind = "bvecs"
        default_fmt = "%d"
    else:
        raise ValueError("Unsupported extension. Use .fvecs or .bvecs")

    n_out = n if args.head is None else min(n, args.head)
    fmt = args.fmt if args.fmt is not None else default_fmt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        # header
        f.write(f"{n_out} {d}\n")

        for start in range(0, n_out, args.chunk):
            end = min(n_out, start + args.chunk)

            if kind == "fvecs":
                # mm is int32 matrix (n, d+1); values are in columns 1..d as float32 bytes
                block_i32 = mm[start:end, 1:]          # int32 view
                block = block_i32.view(np.float32)     # float32 reinterpret
            else:
                # mm is uint8 matrix (n, d+4); values are bytes after first 4 bytes
                block = np.asarray(mm[start:end, 4:])  # uint8

            np.savetxt(f, block, fmt=fmt, delimiter=" ")

if __name__ == "__main__":
    main()
