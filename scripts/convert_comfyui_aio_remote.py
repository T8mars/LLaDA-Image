#!/usr/bin/env python3
"""Stream a pinned Hugging Face LLaDA-Image revision into one ComfyUI checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import urllib.request
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from convert_comfyui_aio import (
    COMPONENT_PREFIXES,
    CONFIG_COMPONENTS,
    COPY_CHUNK_SIZE,
    DTYPE_BYTES,
    FORMAT_VERSION,
    TOKENIZER_KEY,
    TensorSource,
    build_manifest,
    is_known_component_key,
    make_metadata,
    padded_header,
    sha256_file,
    validate_variant,
    verify_checkpoint,
)
from verify_comfyui_shape_contract import read_url, remote_url

MAX_METADATA_FILE_SIZE = 64 * 1024 * 1024


@contextmanager
def output_lock(output: Path):
    lock_path = output.with_name(f"{output.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                f"another conversion is already using output {output}"
            ) from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def read_lock(args: argparse.Namespace) -> tuple[Path, list[dict]]:
    lock_path = args.source_lock
    if lock_path is None:
        lock_path = (
            Path(__file__).resolve().parents[1]
            / "manifests"
            / f"llada-image-{args.variant}.source.json"
        )
    lock_path = lock_path.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    identity = {
        "format_version": FORMAT_VERSION,
        "variant": args.variant,
        "source_repo": args.source_repo,
        "source_revision": args.source_revision,
    }
    for key, expected in identity.items():
        if lock.get(key) != expected:
            raise ValueError(
                f"{lock_path}: expected {key}={expected!r}, got {lock.get(key)!r}"
            )

    entries = lock.get("files")
    if not isinstance(entries, list):
        raise TypeError(f"{lock_path}: files must be a list")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError(f"{lock_path}: malformed file entry")
        path = entry.get("path")
        size = entry.get("size")
        digest = entry.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError(f"{lock_path}: malformed file entry {entry!r}")
        if path in seen:
            raise ValueError(f"{lock_path}: duplicate file entry {path!r}")
        seen.add(path)
    return lock_path, sorted(entries, key=lambda entry: entry["path"])


def verify_bytes(entry: dict, data: bytes) -> None:
    if len(data) != entry["size"]:
        raise ValueError(
            f"{entry['path']}: expected {entry['size']} bytes, got {len(data)}"
        )
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(
            f"{entry['path']}: SHA-256 mismatch; expected {entry['sha256']}, got {digest}"
        )


def download_metadata_files(
    args: argparse.Namespace, entries: list[dict]
) -> dict[str, bytes]:
    files = {}
    for entry in entries:
        if entry["path"].endswith(".safetensors"):
            continue
        if entry["size"] > MAX_METADATA_FILE_SIZE:
            raise ValueError(
                f"{entry['path']}: non-weight file exceeds {MAX_METADATA_FILE_SIZE} bytes"
            )
        data = read_url(
            remote_url(args.source_repo, args.source_revision, entry["path"])
        )
        verify_bytes(entry, data)
        files[entry["path"]] = data
    return files


def json_object(files: dict[str, bytes], path: str) -> dict:
    try:
        value = json.loads(files[path])
    except KeyError as exc:
        raise FileNotFoundError(f"source lock is missing required file {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def build_config(files: dict[str, bytes], variant: str) -> dict:
    config = {
        name: json_object(files, f"{name}/config.json")
        for name in CONFIG_COMPONENTS
    }
    config["scheduler"] = json_object(files, "scheduler/scheduler_config.json")
    config["tokenizer"] = json_object(files, "tokenizer/tokenizer_config.json")
    config["pipeline"] = json_object(files, "model_index.json")
    config["llada_image"] = {
        "format_version": FORMAT_VERSION,
        "variant": variant,
        "recommended_steps": 50 if variant == "base" else 4,
        "recommended_cfg": 5.0 if variant == "base" else 1.0,
        "recommended_sampler": "euler" if variant == "base" else "llada_image_turbo",
        "recommended_scheduler": "llada_image",
    }
    return config


def read_remote_header(
    args: argparse.Namespace, entry: dict
) -> tuple[dict, int]:
    url = remote_url(args.source_repo, args.source_revision, entry["path"])
    length_data = read_url(url, (0, 7))
    if len(length_data) != 8:
        raise ValueError(f"{entry['path']}: truncated safetensors length")
    header_length = struct.unpack("<Q", length_data)[0]
    if header_length < 2 or header_length % 8 or header_length > entry["size"] - 8:
        raise ValueError(
            f"{entry['path']}: invalid safetensors header length {header_length}"
        )
    header_data = read_url(url, (8, 7 + header_length))
    if len(header_data) != header_length:
        raise ValueError(f"{entry['path']}: truncated safetensors header")
    try:
        header = json.loads(header_data.decode("utf-8").rstrip(" "))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{entry['path']}: invalid safetensors header") from exc
    if not isinstance(header, dict):
        raise TypeError(f"{entry['path']}: safetensors header must be an object")
    return header, 8 + header_length


def collect_tensors(
    args: argparse.Namespace, entries: list[dict], metadata_files: dict[str, bytes]
) -> tuple[list[TensorSource], dict[str, dict]]:
    entry_by_path = {entry["path"]: entry for entry in entries}
    tensors = []
    destination_keys = set()
    weight_entries = {}

    for component, prefix in COMPONENT_PREFIXES:
        component_entries = [
            entry
            for entry in entries
            if entry["path"].startswith(f"{component}/")
            and entry["path"].endswith(".safetensors")
        ]
        if not component_entries:
            raise FileNotFoundError(
                f"source lock has no safetensors files for {component}"
            )
        source_keys = {}
        for entry in component_entries:
            header, data_start = read_remote_header(args, entry)
            data_size = entry["size"] - data_start
            file_tensors = []
            for source_key, tensor in header.items():
                if source_key == "__metadata__":
                    continue
                if not isinstance(tensor, dict):
                    raise TypeError(
                        f"{entry['path']}: malformed tensor entry {source_key}"
                    )
                offsets = tensor.get("data_offsets")
                shape = tensor.get("shape")
                dtype = tensor.get("dtype")
                if (
                    not isinstance(offsets, list)
                    or len(offsets) != 2
                    or not all(isinstance(value, int) for value in offsets)
                    or offsets[0] < 0
                    or offsets[1] < offsets[0]
                    or offsets[1] > data_size
                    or not isinstance(shape, list)
                    or not all(
                        isinstance(value, int) and value >= 0 for value in shape
                    )
                    or not isinstance(dtype, str)
                    or dtype not in DTYPE_BYTES
                ):
                    raise ValueError(
                        f"{entry['path']}: malformed tensor metadata for {source_key}"
                    )
                size = math.prod(shape) * DTYPE_BYTES[dtype]
                if size != offsets[1] - offsets[0]:
                    raise ValueError(
                        f"{entry['path']}: tensor byte size does not match dtype/shape for {source_key}"
                    )
                if not is_known_component_key(component, source_key):
                    raise ValueError(
                        f"{entry['path']}: unknown {component} tensor key: {source_key}"
                    )
                if source_key in source_keys:
                    raise ValueError(
                        f"duplicate source tensor key {source_key!r} in "
                        f"{source_keys[source_key]!r} and {entry['path']!r}"
                    )
                destination = f"{prefix}{source_key}"
                if destination in destination_keys:
                    raise ValueError(f"duplicate destination tensor key {destination}")
                source_keys[source_key] = entry["path"]
                destination_keys.add(destination)
                file_tensors.append(
                    TensorSource(
                        key=destination,
                        dtype=dtype,
                        shape=shape,
                        size=size,
                        path=Path(entry["path"]),
                        offset=data_start + offsets[0],
                    )
                )

            file_tensors.sort(key=lambda tensor: tensor.offset)
            cursor = data_start
            for tensor in file_tensors:
                if tensor.offset != cursor:
                    raise ValueError(
                        f"{entry['path']}: tensor data is not contiguous before {tensor.key}"
                    )
                cursor += tensor.size
            if cursor != entry["size"]:
                raise ValueError(
                    f"{entry['path']}: tensor data does not cover the complete file"
                )
            tensors.extend(file_tensors)
            weight_entries[entry["path"]] = entry

        indexes = [
            path
            for path in metadata_files
            if path.startswith(f"{component}/")
            and path.endswith(".safetensors.index.json")
        ]
        if len(indexes) > 1:
            raise ValueError(f"multiple safetensors index files for {component}")
        if indexes:
            index = json_object(metadata_files, indexes[0])
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in weight_map.items()
            ):
                raise ValueError(f"{indexes[0]}: invalid weight_map")
            if set(weight_map) != set(source_keys):
                raise ValueError(f"{indexes[0]}: shard/index key mismatch")
            for key, filename in weight_map.items():
                expected_path = f"{component}/{filename}"
                if source_keys[key] != expected_path or expected_path not in entry_by_path:
                    raise ValueError(
                        f"{indexes[0]}: invalid shard mapping for {key!r}"
                    )

    tokenizer_data = metadata_files.get("tokenizer/tokenizer.json")
    if tokenizer_data is None:
        raise FileNotFoundError("source lock is missing tokenizer/tokenizer.json")
    tensors.append(
        TensorSource(
            key=TOKENIZER_KEY,
            dtype="U8",
            shape=[len(tokenizer_data)],
            size=len(tokenizer_data),
            data=tokenizer_data,
        )
    )
    return tensors, weight_entries


def open_remote(
    args: argparse.Namespace, path: str, start: int, size: int
) -> BinaryIO:
    headers = {
        "User-Agent": "llada-image-comfyui-remote-converter/1",
        "Accept-Encoding": "identity",
    }
    if start:
        headers["Range"] = f"bytes={start}-{size - 1}"
    request = urllib.request.Request(
        remote_url(args.source_repo, args.source_revision, path), headers=headers
    )
    response = urllib.request.urlopen(request, timeout=60)
    if start and getattr(response, "status", 206) != 206:
        response.close()
        raise RuntimeError(f"server did not honor resume range for {path}")
    return response


def stream_weight_file(
    args: argparse.Namespace,
    path: str,
    entry: dict,
    data_start: int,
    target: BinaryIO,
    output_digest: hashlib._Hash,
) -> None:
    source_digest = hashlib.sha256()
    cursor = 0
    failures = 0
    report_interval = 512 * 1024 * 1024
    next_report = report_interval
    while cursor < entry["size"]:
        try:
            with open_remote(args, path, cursor, entry["size"]) as source:
                while cursor < entry["size"]:
                    chunk_start = cursor
                    data = source.read(
                        min(COPY_CHUNK_SIZE, entry["size"] - cursor)
                    )
                    if not data:
                        raise EOFError(
                            f"{path}: remote stream ended at {cursor} of "
                            f"{entry['size']} bytes"
                        )
                    source_digest.update(data)
                    cursor += len(data)
                    payload_start = max(chunk_start, data_start)
                    if cursor > payload_start:
                        payload = data[payload_start - chunk_start :]
                        target.write(payload)
                        output_digest.update(payload)
                    if cursor >= next_report or cursor == entry["size"]:
                        print(
                            f"  {path}: {cursor}/{entry['size']} bytes",
                            flush=True,
                        )
                        next_report = cursor + report_interval
                if source.read(1):
                    raise ValueError(f"{path}: remote file exceeds locked size")
        except (EOFError, OSError, RuntimeError) as exc:
            failures += 1
            if failures >= args.retries:
                raise RuntimeError(
                    f"{path}: failed after {failures} interrupted streams"
                ) from exc
            print(
                f"Resuming {path} at byte {cursor} after stream failure "
                f"{failures}/{args.retries}: {exc}",
                flush=True,
            )
    digest = source_digest.hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(
            f"{path}: SHA-256 mismatch; expected {entry['sha256']}, got {digest}"
        )


def write_checkpoint(
    args: argparse.Namespace,
    tensors: list[TensorSource],
    weight_entries: dict[str, dict],
    metadata: dict[str, str],
) -> str:
    header = padded_header(tensors, metadata)
    output_digest = hashlib.sha256()
    partial = args.output.with_name(f"{args.output.name}.partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    by_path = defaultdict(list)
    for tensor in tensors:
        if tensor.path is not None:
            by_path[tensor.path.as_posix()].append(tensor)

    try:
        with partial.open("wb") as target:
            prefix = struct.pack("<Q", len(header)) + header
            target.write(prefix)
            output_digest.update(prefix)
            for path, path_tensors in by_path.items():
                entry = weight_entries[path]
                data_start = path_tensors[0].offset
                print(f"Streaming {path} ({entry['size']} bytes)", flush=True)
                stream_weight_file(
                    args,
                    path,
                    entry,
                    data_start,
                    target,
                    output_digest,
                )

            tokenizer = tensors[-1]
            if tokenizer.key != TOKENIZER_KEY or tokenizer.data is None:
                raise AssertionError("tokenizer tensor must be the final conversion item")
            target.write(tokenizer.data)
            output_digest.update(tokenizer.data)
        os.replace(partial, args.output)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return output_digest.hexdigest()


def convert_locked(args: argparse.Namespace) -> None:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {args.output}; pass --overwrite to replace it"
        )
    lock_path, entries = read_lock(args)
    metadata_files = download_metadata_files(args, entries)
    config = build_config(metadata_files, args.variant)
    validate_variant(config, args.variant)
    tensors, weight_entries = collect_tensors(args, entries, metadata_files)
    metadata = make_metadata(config, args, sha256_file(lock_path), tensors)

    print(f"Packing {len(tensors)} tensors into {args.output}")
    output_sha256 = write_checkpoint(
        args, tensors, weight_entries, metadata
    )
    verify_checkpoint(args.output, tensors, metadata, output_sha256)
    manifest = build_manifest(args.output, tensors, entries, output_sha256, args)
    manifest_path = args.output.with_suffix(f"{args.output.suffix}.manifest.json")
    partial = manifest_path.with_name(f"{manifest_path.name}.partial")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(partial, manifest_path)
    print(
        f"Wrote {args.output} ({args.output.stat().st_size} bytes, "
        f"sha256 {output_sha256})"
    )
    print(f"Wrote {manifest_path}")


def convert(args: argparse.Namespace) -> None:
    args.output = args.output.resolve()
    with output_lock(args.output):
        convert_locked(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Destination AIO .safetensors file")
    parser.add_argument("--variant", choices=("base", "turbo"), required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.retries < 1:
        parser.error("--retries must be at least 1")
    return args


if __name__ == "__main__":
    convert(parse_args())
