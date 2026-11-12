"""Package Standard Regenerate LoRA + Sharp curves + reference table unchanged.

This self-contained checkpoint is for TS H3 Regenerate Model, not a generic LoRA
loader. No dtype conversion, Turbo replacement or factor recompression is done.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

TABLE_HASH = "c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7"
TABLE_KEY = "ts_regenerate.adaln_t_table"


def header(path):
    with path.open("rb") as stream:
        length, = struct.unpack("<Q", stream.read(8))
        if not 0 < length < 100_000_000:
            raise ValueError(f"Invalid safetensors header: {path}")
        return json.loads(stream.read(length)), 8 + length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lora", "sharp", "reference", "output", "manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sources = [args.lora.resolve(strict=True), args.sharp.resolve(strict=True), args.reference.resolve(strict=True)]
    output, manifest = args.output.resolve(), args.manifest.resolve()
    part = output.with_suffix(output.suffix + ".part")
    if output in sources or output.exists() or part.exists() or manifest.exists():
        raise FileExistsError("Output, .part and manifest must not exist; sources are read-only.")
    headers = [header(path) for path in sources]
    main_keys = {k for k in headers[0][0] if k != "__metadata__"}
    names = {k.removesuffix(".lora_A.weight") for k in main_keys if k.endswith(".lora_A.weight")}
    if len(names) != 312 or main_keys != {n + s for n in names for s in (".lora_A.weight", ".lora_B.weight")}:
        raise ValueError("Expected the original 312-pair Standard Regenerate adapter.")
    if any(headers[0][0][n + ".lora_A.weight"]["shape"][0] != 576 for n in names):
        raise ValueError("Expected rank576: Refiner384 + Turbo128 + Sharp64.")
    sharp_keys = {f"transformer.transformer_blocks.{i}.adaln_proj.linear.curve_{ab}.weight"
                  for i in range(50) for ab in ("A", "B")}
    if {k for k in headers[1][0] if k != "__metadata__"} != sharp_keys:
        raise ValueError("Expected the 50 original Sharp curve A/B pairs.")
    tables = [k for k in headers[2][0] if k.endswith("adaln_t_table")]
    if len(tables) != 1:
        raise ValueError("Reference checkpoint must contain exactly one AdaLN table.")
    info = headers[2][0][tables[0]]
    if info["shape"] != [1025, 8] or info["dtype"] != "F32":
        raise ValueError("Reference table must be 1025x8 FP32.")
    with sources[2].open("rb") as source:
        start, end = info["data_offsets"]
        source.seek(headers[2][1] + start)
        if hashlib.sha256(source.read(end - start)).hexdigest() != TABLE_HASH:
            raise ValueError("Reference table hash differs from the released Sharp table.")
    plan = [(key, index, key, headers[index][0][key])
            for index, keys in ((0, main_keys), (1, sharp_keys)) for key in sorted(keys)]
    plan.append((TABLE_KEY, 2, tables[0], info))
    metadata = {"format": "pt", "ts_h3_regenerate_bundle": "standard-v1",
                "ts_h3_regenerate_turbo": "lightx2v/Minimax-h3-Turbo/minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors",
                "ts_h3_regenerate_payloads": "Original FP32 factors and reference table, copied bitwise"}
    target, offset = {"__metadata__": metadata}, 0
    for key, _, _, value in plan:
        size = value["data_offsets"][1] - value["data_offsets"][0]
        target[key] = dict(value, data_offsets=[offset, offset + size])
        offset += size
    encoded = json.dumps(target, separators=(",", ":")).encode("utf8")
    encoded += b" " * (-len(encoded) % 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with part.open("xb") as dest:
        dest.write(struct.pack("<Q", len(encoded)))
        dest.write(encoded)
        for key, index, source_key, value in plan:
            start, end = value["data_offsets"]
            digest = hashlib.sha256()
            with sources[index].open("rb") as source:
                source.seek(headers[index][1] + start)
                remaining = end - start
                while remaining:
                    data = source.read(min(16 * 1024**2, remaining))
                    if not data:
                        raise EOFError(source_key)
                    dest.write(data)
                    digest.update(data)
                    remaining -= len(data)
            records.append({"key": key, "source": index, "source_key": source_key,
                            "bytes": end - start, "sha256": digest.hexdigest()})
    with part.open("rb") as check:
        check.seek(8 + len(encoded))
        for record in records:
            digest, remaining = hashlib.sha256(), record["bytes"]
            while remaining:
                data = check.read(min(16 * 1024**2, remaining))
                if not data:
                    raise EOFError(record["key"])
                digest.update(data)
                remaining -= len(data)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError(f"Output differs at {record['key']}")
        assert not check.read(1)
    part.rename(output)
    report = {"output": str(output), "bytes": output.stat().st_size, "version": "standard-v1",
              "sources": [str(p) for p in sources], "bitwise_verified_tensors": len(records),
              "turbo": metadata["ts_h3_regenerate_turbo"], "tensors": records}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, indent=2), encoding="utf8")
    sys.stdout.write(json.dumps({k: v for k, v in report.items() if k != "tensors"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
