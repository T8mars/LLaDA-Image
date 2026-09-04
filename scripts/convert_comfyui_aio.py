#!/usr/bin/env python3
"""Pack an official LLaDA-Image repository into one ComfyUI checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

COMPONENT_PREFIXES = (
    ("transformer", "model.diffusion_model."),
    ("text_encoder", "text_encoders.llada2."),
    ("queryformer", "text_encoders.queryformer."),
    ("text_projection", "text_encoders.text_projection."),
    ("sigvq", "text_encoders.sigvq."),
    ("vae", "vae."),
)
CONFIG_COMPONENTS = tuple(name for name, _ in COMPONENT_PREFIXES)
TOKENIZER_KEY = "text_encoders.tokenizer_json"
COPY_CHUNK_SIZE = 16 * 1024 * 1024
FORMAT_VERSION = "1"
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
COMPONENT_KEY_ROOTS = {
    "transformer": (
        "all_final_layer.",
        "all_x_embedder.",
        "cap_embedder.",
        "cap_pad_token",
        "context_refiner.",
        "layers.",
        "noise_refiner.",
        "semantic_embedder.",
        "sigvq_embedder.",
        "sigvq_pad_token",
        "sigvq_refiner.",
        "t_embedder.",
        "x_pad_token",
    ),
    "text_encoder": ("model.",),
    "queryformer": ("meta_queries", "query_blocks."),
    "text_projection": ("layers.", "projector."),
    "sigvq": ("prior_projector.", "prior_token_embedding.", "visual.", "vqmodel."),
    "vae": ("bn.", "decoder.", "encoder.", "post_quant_conv.", "quant_conv."),
}


def is_known_component_key(component: str, key: str) -> bool:
    return any(
        key.startswith(root) if root.endswith(".") else key == root
        for root in COMPONENT_KEY_ROOTS[component]
    )


@dataclass(frozen=True)
class TensorSource:
    key: str
    dtype: str
    shape: list[int]
    size: int
    path: Path | None = None
    offset: int = 0
    data: bytes | None = None


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        length_data = handle.read(8)
        if len(length_data) != 8:
            raise ValueError(f"{path}: truncated safetensors length")
        header_length = struct.unpack("<Q", length_data)[0]
        if header_length < 2 or header_length > path.stat().st_size - 8:
            raise ValueError(
                f"{path}: invalid safetensors header length {header_length}"
            )
        if header_length % 8:
            raise ValueError(
                f"{path}: safetensors header length must be aligned to 8 bytes"
            )
        header_data = handle.read(header_length)
    try:
        header = json.loads(header_data.decode("utf-8").rstrip(" "))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid safetensors header") from exc
    if not isinstance(header, dict):
        raise TypeError(f"{path}: safetensors header must be an object")
    return header, 8 + header_length


def collect_component(
    root: Path, component: str, prefix: str
) -> tuple[list[TensorSource], list[Path]]:
    directory = root / component
    if not directory.is_dir():
        raise FileNotFoundError(f"missing required component directory: {directory}")
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files found in {directory}")

    tensors = []
    source_keys: dict[str, Path] = {}
    for path in files:
        header, data_start = read_safetensors_header(path)
        data_size = path.stat().st_size - data_start
        data_ranges = []
        for source_key in sorted(key for key in header if key != "__metadata__"):
            entry = header[source_key]
            if not isinstance(entry, dict):
                raise TypeError(f"{path}: malformed tensor entry {source_key}")
            offsets = entry.get("data_offsets")
            shape = entry.get("shape")
            dtype = entry.get("dtype")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) for value in offsets)
                or offsets[0] < 0
                or offsets[1] < offsets[0]
                or offsets[1] > data_size
                or not isinstance(shape, list)
                or not all(isinstance(value, int) and value >= 0 for value in shape)
                or not isinstance(dtype, str)
            ):
                raise ValueError(f"{path}: malformed tensor metadata for {source_key}")
            if dtype not in DTYPE_BYTES:
                raise ValueError(
                    f"{path}: unsupported safetensors dtype {dtype!r} for {source_key}"
                )
            tensor_size = math.prod(shape) * DTYPE_BYTES[dtype]
            if tensor_size != offsets[1] - offsets[0]:
                raise ValueError(
                    f"{path}: tensor byte size does not match dtype/shape for {source_key}: "
                    f"expected {tensor_size}, got {offsets[1] - offsets[0]}"
                )
            if not is_known_component_key(component, source_key):
                raise ValueError(
                    f"{path}: unknown {component} tensor key: {source_key}"
                )
            if source_key in source_keys:
                raise ValueError(
                    f"duplicate source tensor key {source_key!r} in {source_keys[source_key]} and {path}"
                )
            source_keys[source_key] = path
            data_ranges.append((offsets[0], offsets[1], source_key))
            tensors.append(
                TensorSource(
                    key=f"{prefix}{source_key}",
                    dtype=dtype,
                    shape=shape,
                    size=offsets[1] - offsets[0],
                    path=path,
                    offset=data_start + offsets[0],
                )
            )
        cursor = 0
        for start, end, source_key in sorted(data_ranges):
            if start != cursor:
                raise ValueError(
                    f"{path}: tensor data is not contiguous before {source_key}; "
                    f"expected offset {cursor}, got {start}"
                )
            cursor = end
        if cursor != data_size:
            raise ValueError(
                f"{path}: {data_size - cursor} trailing tensor-data bytes are not declared"
            )
    indexes = sorted(directory.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise ValueError(f"{directory}: multiple safetensors index files")
    if indexes:
        index = read_json(indexes[0])
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in weight_map.items()
        ):
            raise ValueError(f"{indexes[0]}: invalid weight_map")
        if set(weight_map) != set(source_keys):
            missing = sorted(set(weight_map) - set(source_keys))
            unknown = sorted(set(source_keys) - set(weight_map))
            raise ValueError(
                f"{indexes[0]}: shard/index key mismatch; missing={missing[:5]}, unknown={unknown[:5]}"
            )
        for key, filename in weight_map.items():
            if source_keys[key].name != filename:
                raise ValueError(
                    f"{indexes[0]}: {key!r} is indexed in {filename!r}, found in {source_keys[key].name!r}"
                )
        files.append(indexes[0])
    return tensors, files


def read_json(path: Path, *, required: bool = True) -> dict | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"missing required JSON file: {path}")
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def build_config(root: Path, variant: str) -> dict:
    config = {
        name: read_json(root / name / "config.json") for name in CONFIG_COMPONENTS
    }
    config["scheduler"] = read_json(root / "scheduler" / "scheduler_config.json")
    config["tokenizer"] = (
        read_json(root / "tokenizer" / "tokenizer_config.json", required=False) or {}
    )
    config["pipeline"] = read_json(root / "model_index.json")
    config["llada_image"] = {
        "format_version": FORMAT_VERSION,
        "variant": variant,
        "recommended_steps": 50 if variant == "base" else 4,
        "recommended_cfg": 5.0 if variant == "base" else 1.0,
        "recommended_sampler": "euler" if variant == "base" else "llada_image_turbo",
        "recommended_scheduler": "llada_image",
    }
    return config


def validate_variant(config: dict, variant: str) -> None:
    scheduler = config["scheduler"]
    expected = {
        "base": {
            "shift": 1.0,
            "stochastic_sampling": False,
            "use_uniform_sigmas": False,
        },
        "turbo": {
            "shift": 3.0,
            "stochastic_sampling": True,
            "use_uniform_sigmas": True,
        },
    }[variant]
    for key, value in expected.items():
        actual = scheduler.get(key, False)
        if actual != value:
            raise ValueError(
                f"scheduler config does not match --variant {variant}: expected {key}={value!r}, got {actual!r}"
            )


def make_metadata(
    config: dict,
    args: argparse.Namespace,
    source_lock_sha256: str,
    tensors: list[TensorSource],
) -> dict[str, str]:
    return {
        "config": json.dumps(config, ensure_ascii=False, separators=(",", ":")),
        "modelspec.sai_model_spec": "1.0.0",
        "modelspec.architecture": "llada-image",
        "modelspec.implementation": "comfy",
        "modelspec.title": f"LLaDA-Image {args.variant.title()} AIO",
        "modelspec.description": "Native ComfyUI AIO checkpoint for LLaDA-Image",
        "modelspec.license": "Apache-2.0",
        "llada_image.format_version": FORMAT_VERSION,
        "llada_image.variant": args.variant,
        "llada_image.source_repo": args.source_repo,
        "llada_image.source_revision": args.source_revision,
        "llada_image.source_lock_sha256": source_lock_sha256,
        "llada_image.weight_dtypes": json.dumps(
            sorted({tensor.dtype for tensor in tensors if tensor.key != TOKENIZER_KEY})
        ),
        "llada_image.converter": "scripts/convert_comfyui_aio.py",
    }


def padded_header(tensors: list[TensorSource], metadata: dict[str, str]) -> bytes:
    header: dict[str, object] = {"__metadata__": metadata}
    offset = 0
    for tensor in tensors:
        header[tensor.key] = {
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [offset, offset + tensor.size],
        }
        offset += tensor.size
    encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return encoded + b" " * ((-len(encoded)) % 8)


def copy_range(
    source: BinaryIO, target: BinaryIO, size: int, digest: hashlib._Hash
) -> None:
    remaining = size
    while remaining:
        data = source.read(min(remaining, COPY_CHUNK_SIZE))
        if not data:
            raise EOFError("source safetensors ended inside a tensor")
        target.write(data)
        digest.update(data)
        remaining -= len(data)


def write_checkpoint(
    output: Path, tensors: list[TensorSource], metadata: dict[str, str]
) -> str:
    header = padded_header(tensors, metadata)
    digest = hashlib.sha256()
    partial = output.with_name(f"{output.name}.partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    try:
        with partial.open("wb") as target:
            prefix = struct.pack("<Q", len(header)) + header
            target.write(prefix)
            digest.update(prefix)

            current_path = None
            source = None
            try:
                for tensor in tensors:
                    if tensor.data is not None:
                        target.write(tensor.data)
                        digest.update(tensor.data)
                        continue
                    if tensor.path != current_path:
                        if source is not None:
                            source.close()
                        current_path = tensor.path
                        source = current_path.open("rb")
                    source.seek(tensor.offset)
                    copy_range(source, target, tensor.size, digest)
            finally:
                if source is not None:
                    source.close()
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(COPY_CHUNK_SIZE):
            digest.update(data)
    return digest.hexdigest()


def build_manifest(
    output: Path,
    tensors: list[TensorSource],
    sources: list[dict],
    output_sha256: str,
    args: argparse.Namespace,
) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "variant": args.variant,
        "source_repo": args.source_repo,
        "source_revision": args.source_revision,
        "output": output.name,
        "output_size": output.stat().st_size,
        "output_sha256": output_sha256,
        "tensor_count": len(tensors),
        "sources": sources,
    }


def verify_checkpoint(
    path: Path,
    expected: list[TensorSource],
    expected_metadata: dict[str, str],
    expected_sha256: str,
) -> None:
    header, data_start = read_safetensors_header(path)
    actual_keys = [key for key in header if key != "__metadata__"]
    expected_keys = [tensor.key for tensor in expected]
    if actual_keys != expected_keys:
        raise ValueError(
            "written checkpoint key order does not match the conversion plan"
        )
    expected_size = sum(tensor.size for tensor in expected)
    if path.stat().st_size - data_start != expected_size:
        raise ValueError(
            "written checkpoint data length does not match tensor metadata"
        )
    if header.get("__metadata__") != expected_metadata:
        raise ValueError(
            "written checkpoint metadata does not match the conversion plan"
        )
    for tensor in expected:
        entry = header[tensor.key]
        if entry["dtype"] != tensor.dtype or entry["shape"] != tensor.shape:
            raise ValueError(f"written checkpoint metadata mismatch for {tensor.key}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"written checkpoint hash mismatch: expected {expected_sha256}, got {actual_sha256}"
        )


def metadata_source_files(root: Path) -> list[Path]:
    paths = [root / component / "config.json" for component in CONFIG_COMPONENTS]
    paths.extend(
        (root / "scheduler" / "scheduler_config.json", root / "model_index.json")
    )
    tokenizer_config = root / "tokenizer" / "tokenizer_config.json"
    if tokenizer_config.is_file():
        paths.append(tokenizer_config)
    return paths


def verify_source_lock(
    root: Path, source_files: list[Path], args: argparse.Namespace
) -> tuple[list[dict], str]:
    lock_path = args.source_lock
    if lock_path is None:
        lock_path = (
            Path(__file__).resolve().parents[1]
            / "manifests"
            / f"llada-image-{args.variant}.source.json"
        )
    lock_path = lock_path.resolve()
    lock = read_json(lock_path)
    identity = {
        "format_version": FORMAT_VERSION,
        "variant": args.variant,
        "source_repo": args.source_repo,
        "source_revision": args.source_revision,
    }
    for key, value in identity.items():
        if lock.get(key) != value:
            raise ValueError(
                f"{lock_path}: expected {key}={value!r}, got {lock.get(key)!r}"
            )
    entries = lock.get("files")
    if not isinstance(entries, list):
        raise TypeError(f"{lock_path}: files must be a list")
    expected = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError(f"{lock_path}: malformed file entry")
        path = entry.get("path")
        size = entry.get("size")
        digest = entry.get("sha256")
        if (
            not isinstance(path, str)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError(f"{lock_path}: malformed file entry {entry!r}")
        if path in expected:
            raise ValueError(f"{lock_path}: duplicate file entry {path!r}")
        expected[path] = entry

    paths = {path.resolve() for path in source_files}
    actual_names = {path.relative_to(root).as_posix() for path in paths}
    if actual_names != set(expected):
        missing = sorted(set(expected) - actual_names)
        unknown = sorted(actual_names - set(expected))
        raise ValueError(
            f"{lock_path}: source file set mismatch; missing={missing[:5]}, unknown={unknown[:5]}"
        )

    verified = []
    for name in sorted(actual_names):
        path = root / Path(name)
        entry = expected[name]
        size = path.stat().st_size
        if size != entry["size"]:
            raise ValueError(f"{path}: expected {entry['size']} bytes, got {size}")
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError(
                f"{path}: SHA-256 mismatch; expected {entry['sha256']}, got {digest}"
            )
        verified.append({"path": name, "size": size, "sha256": digest})
    return verified, sha256_file(lock_path)


def convert(args: argparse.Namespace) -> None:
    root = args.input.resolve()
    output = args.output.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {root}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {output}; pass --overwrite to replace it"
        )

    config = build_config(root, args.variant)
    validate_variant(config, args.variant)

    tensors = []
    source_files = []
    for component, prefix in COMPONENT_PREFIXES:
        component_tensors, component_files = collect_component(root, component, prefix)
        tensors.extend(component_tensors)
        source_files.extend(component_files)
    source_files.extend(metadata_source_files(root))

    tokenizer_path = root / "tokenizer" / "tokenizer.json"
    tokenizer_data = tokenizer_path.read_bytes()
    source_files.append(tokenizer_path)
    tensors.append(
        TensorSource(
            key=TOKENIZER_KEY,
            dtype="U8",
            shape=[len(tokenizer_data)],
            size=len(tokenizer_data),
            data=tokenizer_data,
        )
    )

    seen = set()
    for tensor in tensors:
        if tensor.key in seen:
            raise ValueError(f"duplicate destination tensor key: {tensor.key}")
        seen.add(tensor.key)

    verified_sources, source_lock_sha256 = verify_source_lock(root, source_files, args)
    metadata = make_metadata(config, args, source_lock_sha256, tensors)
    print(f"Packing {len(tensors)} tensors into {output}")
    output_sha256 = write_checkpoint(output, tensors, metadata)
    verify_checkpoint(output, tensors, metadata, output_sha256)

    manifest = build_manifest(output, tensors, verified_sources, output_sha256, args)
    manifest_path = output.with_suffix(f"{output.suffix}.manifest.json")
    manifest_partial = manifest_path.with_name(f"{manifest_path.name}.partial")
    with manifest_partial.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(manifest_partial, manifest_path)
    print(f"Wrote {output} ({output.stat().st_size} bytes, sha256 {output_sha256})")
    print(f"Wrote {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="Downloaded Hugging Face model directory"
    )
    parser.add_argument("output", type=Path, help="Destination AIO .safetensors file")
    parser.add_argument("--variant", choices=("base", "turbo"), required=True)
    parser.add_argument(
        "--source-repo", required=True, help="For example inclusionAI/LLaDA-Image"
    )
    parser.add_argument(
        "--source-revision", required=True, help="Exact Hugging Face commit revision"
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        help="Pinned source manifest; defaults to manifests/llada-image-<variant>.source.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
