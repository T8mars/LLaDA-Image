#!/usr/bin/env python3
"""Stream a pinned Hugging Face LLaDA-Image revision into one ComfyUI checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import time
import urllib.error
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
    canonical_json_sha256,
    is_known_component_key,
    make_metadata,
    padded_header,
    validate_variant,
    verify_checkpoint,
)
from verify_comfyui_shape_contract import read_url, remote_url

MAX_METADATA_FILE_SIZE = 64 * 1024 * 1024
PARTIAL_CHECKPOINT_INTERVAL = 64 * 1024 * 1024
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


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


def read_lock(args: argparse.Namespace) -> tuple[dict, list[dict]]:
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
    return lock, sorted(entries, key=lambda entry: entry["path"])


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


def read_remote_bytes(
    args: argparse.Namespace,
    path: str,
    byte_range: tuple[int, int] | None = None,
) -> bytes:
    url = remote_url(args.source_repo, args.source_revision, path)
    retries = getattr(args, "retries", 20)
    error = None
    for attempt in range(1, retries + 1):
        try:
            return read_url(url, byte_range)
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_HTTP_CODES:
                raise
            error = exc
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            delay = int(retry_after) if retry_after and retry_after.isdigit() else 0
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            error = exc
            delay = 0
        if attempt == retries:
            break
        delay = max(delay, min(2 ** (attempt - 1), 30))
        print(
            f"Retrying {path} metadata/header request after attempt "
            f"{attempt}/{retries} in {delay}s: {error}",
            flush=True,
        )
        time.sleep(delay)
    raise RuntimeError(
        f"{path}: metadata/header request failed after {retries} attempts"
    ) from error


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
        data = read_remote_bytes(args, entry["path"])
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
    length_data = read_remote_bytes(args, entry["path"], (0, 7))
    if len(length_data) != 8:
        raise ValueError(f"{entry['path']}: truncated safetensors length")
    header_length = struct.unpack("<Q", length_data)[0]
    if header_length < 2 or header_length % 8 or header_length > entry["size"] - 8:
        raise ValueError(
            f"{entry['path']}: invalid safetensors header length {header_length}"
        )
    header_data = read_remote_bytes(
        args, entry["path"], (8, 7 + header_length)
    )
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
    cursor: int,
    target: BinaryIO,
    source_digest: hashlib._Hash,
    output_digest: hashlib._Hash,
    checkpoint,
) -> None:
    failures = 0
    report_interval = 512 * 1024 * 1024
    next_report = ((cursor // report_interval) + 1) * report_interval
    next_checkpoint = (
        ((cursor // PARTIAL_CHECKPOINT_INTERVAL) + 1)
        * PARTIAL_CHECKPOINT_INTERVAL
    )
    while cursor < entry["size"]:
        try:
            with open_remote(args, path, cursor, entry["size"]) as source:
                while cursor < entry["size"]:
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
                    target.write(data)
                    output_digest.update(data)
                    if cursor >= next_checkpoint:
                        checkpoint(cursor, source_digest.hexdigest())
                        next_checkpoint = (
                            (cursor // PARTIAL_CHECKPOINT_INTERVAL) + 1
                        ) * PARTIAL_CHECKPOINT_INTERVAL
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


def partial_state_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.partial.json")


def write_partial_state(path: Path, state: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(state, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def checkpoint_resume_state(
    args: argparse.Namespace,
    partial: Path,
    state_path: Path,
    prefix: bytes,
    paths: list[str],
    payload_sizes: dict[str, int],
    source_lock_sha256: str,
) -> tuple[dict, int, int]:
    identity = {
        "format_version": FORMAT_VERSION,
        "variant": args.variant,
        "source_repo": args.source_repo,
        "source_revision": args.source_revision,
        "source_lock_sha256": source_lock_sha256,
        "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
    }
    fresh_state = {
        **identity,
        "completed_files": [],
        "current_file": None,
        "current_payload_bytes": 0,
        "current_source_sha256": None,
    }
    if not partial.exists() and not state_path.exists():
        return fresh_state, 0, 0
    if args.overwrite:
        partial.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        return fresh_state, 0, 0
    if not partial.exists() or not state_path.exists():
        if partial.exists() and partial.stat().st_size <= len(prefix):
            partial.unlink()
            state_path.unlink(missing_ok=True)
            return fresh_state, 0, 0
        raise RuntimeError(
            f"incomplete resume pair for {partial}; pass --overwrite to restart"
        )

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid partial resume state: {state_path}") from exc
    if not isinstance(state, dict):
        raise TypeError(f"invalid partial resume state: {state_path}")
    for key, expected in identity.items():
        if state.get(key) != expected:
            raise RuntimeError(
                f"partial resume state mismatch for {key}; pass --overwrite to restart"
            )
    completed = state.get("completed_files")
    if not isinstance(completed, list) or completed != paths[: len(completed)]:
        raise RuntimeError(f"invalid completed-file prefix in {state_path}")

    completed_boundary = len(prefix) + sum(
        payload_sizes[path] for path in completed
    )
    current_file = state.get("current_file")
    current_payload_bytes = state.get("current_payload_bytes", 0)
    current_source_sha256 = state.get("current_source_sha256")
    expected_current_file = paths[len(completed)] if len(completed) < len(paths) else None
    if current_file is None:
        if current_payload_bytes != 0 or current_source_sha256 is not None:
            raise RuntimeError(f"invalid current-file state in {state_path}")
    elif (
        current_file != expected_current_file
        or not isinstance(current_payload_bytes, int)
        or not 0 < current_payload_bytes <= payload_sizes[current_file]
        or not isinstance(current_source_sha256, str)
        or len(current_source_sha256) != 64
    ):
        raise RuntimeError(f"invalid current-file state in {state_path}")

    boundary = completed_boundary + current_payload_bytes
    actual_size = partial.stat().st_size
    if actual_size < boundary:
        raise RuntimeError(
            f"partial checkpoint is shorter than its verified boundary: "
            f"{actual_size} < {boundary}"
        )
    remaining_payload_size = (
        payload_sizes[expected_current_file] - current_payload_bytes
        if expected_current_file is not None
        else None
    )
    if (
        remaining_payload_size is not None
        and actual_size > boundary + remaining_payload_size
    ):
        raise RuntimeError(
            f"partial checkpoint exceeds its next recoverable boundary: "
            f"{actual_size} > {boundary + remaining_payload_size}"
        )
    with partial.open("rb") as handle:
        if handle.read(len(prefix)) != prefix:
            raise RuntimeError(
                "partial checkpoint header mismatch; pass --overwrite to restart"
            )
    return state, len(completed), boundary


def update_digest_from_prefix(
    handle: BinaryIO, digest: hashlib._Hash, size: int
) -> None:
    handle.seek(0)
    remaining = size
    while remaining:
        data = handle.read(min(COPY_CHUNK_SIZE, remaining))
        if not data:
            raise EOFError("partial checkpoint ended before its verified boundary")
        digest.update(data)
        remaining -= len(data)


def source_digest_from_partial(
    args: argparse.Namespace,
    path: str,
    data_start: int,
    target: BinaryIO,
    output_start: int,
    payload_bytes: int,
    expected_digest: str | None,
) -> hashlib._Hash:
    source_prefix = read_remote_bytes(args, path, (0, data_start - 1))
    if len(source_prefix) != data_start:
        raise ValueError(
            f"{path}: expected {data_start} source header bytes, "
            f"got {len(source_prefix)}"
        )
    digest = hashlib.sha256(source_prefix)
    target.seek(output_start)
    remaining = payload_bytes
    while remaining:
        data = target.read(min(COPY_CHUNK_SIZE, remaining))
        if not data:
            raise EOFError(f"partial checkpoint ended inside {path}")
        digest.update(data)
        remaining -= len(data)
    if expected_digest is not None and digest.hexdigest() != expected_digest:
        raise RuntimeError(
            f"partial source digest mismatch for {path}; pass --overwrite to restart"
        )
    target.seek(output_start + payload_bytes)
    return digest


def write_checkpoint(
    args: argparse.Namespace,
    tensors: list[TensorSource],
    weight_entries: dict[str, dict],
    metadata: dict[str, str],
) -> str:
    header = padded_header(tensors, metadata)
    output_digest = hashlib.sha256()
    partial = args.output.with_name(f"{args.output.name}.partial")
    state_path = partial_state_path(args.output)
    partial.parent.mkdir(parents=True, exist_ok=True)
    by_path = defaultdict(list)
    for tensor in tensors:
        if tensor.path is not None:
            by_path[tensor.path.as_posix()].append(tensor)

    prefix = struct.pack("<Q", len(header)) + header
    paths = list(by_path)
    payload_sizes = {
        path: sum(tensor.size for tensor in path_tensors)
        for path, path_tensors in by_path.items()
    }
    state, completed_count, resume_boundary = checkpoint_resume_state(
        args,
        partial,
        state_path,
        prefix,
        paths,
        payload_sizes,
        metadata["llada_image.source_lock_sha256"],
    )

    try:
        mode = "r+b" if resume_boundary else "w+b"
        with partial.open(mode) as target:
            if resume_boundary:
                update_digest_from_prefix(target, output_digest, resume_boundary)
                target.seek(resume_boundary)
                target.truncate()
                print(
                    f"Resuming AIO after {completed_count}/{len(paths)} verified "
                    f"source files at output byte {resume_boundary}",
                    flush=True,
                )
            else:
                target.seek(0)
                target.truncate()
                target.write(prefix)
                target.flush()
                os.fsync(target.fileno())
                write_partial_state(state_path, state)
                output_digest.update(prefix)
            output_start = len(prefix) + sum(
                payload_sizes[path] for path in paths[:completed_count]
            )
            for path in paths[completed_count:]:
                path_tensors = by_path[path]
                entry = weight_entries[path]
                data_start = path_tensors[0].offset
                resume_payload_bytes = (
                    state["current_payload_bytes"]
                    if state.get("current_file") == path
                    else 0
                )
                source_digest = source_digest_from_partial(
                    args,
                    path,
                    data_start,
                    target,
                    output_start,
                    resume_payload_bytes,
                    state.get("current_source_sha256")
                    if resume_payload_bytes
                    else None,
                )

                def checkpoint(
                    source_cursor,
                    source_sha256,
                    current_path=path,
                    source_data_start=data_start,
                ):
                    target.flush()
                    os.fsync(target.fileno())
                    state["current_file"] = current_path
                    state["current_payload_bytes"] = (
                        source_cursor - source_data_start
                    )
                    state["current_source_sha256"] = source_sha256
                    write_partial_state(state_path, state)

                print(f"Streaming {path} ({entry['size']} bytes)", flush=True)
                stream_weight_file(
                    args,
                    path,
                    entry,
                    data_start + resume_payload_bytes,
                    target,
                    source_digest,
                    output_digest,
                    checkpoint,
                )
                target.flush()
                os.fsync(target.fileno())
                state["completed_files"].append(path)
                state["current_file"] = None
                state["current_payload_bytes"] = 0
                state["current_source_sha256"] = None
                write_partial_state(state_path, state)
                output_start += payload_sizes[path]

            tokenizer = tensors[-1]
            if tokenizer.key != TOKENIZER_KEY or tokenizer.data is None:
                raise AssertionError("tokenizer tensor must be the final conversion item")
            target.write(tokenizer.data)
            output_digest.update(tokenizer.data)
        os.replace(partial, args.output)
        state_path.unlink(missing_ok=True)
    except BaseException:
        if partial.exists():
            print(
                f"Kept resumable partial checkpoint {partial} and state "
                f"{state_path}",
                flush=True,
            )
        raise
    return output_digest.hexdigest()


def build_plan(
    args: argparse.Namespace,
    tensors: list[TensorSource],
    metadata: dict[str, str],
) -> dict:
    header = padded_header(tensors, metadata)
    prefix = struct.pack("<Q", len(header)) + header
    offset = 0
    tensor_entries = []
    for tensor in tensors:
        tensor_entries.append(
            {
                "key": tensor.key,
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "size": tensor.size,
                "output_data_offsets": [offset, offset + tensor.size],
                "source_file": tensor.path.as_posix() if tensor.path else None,
                "source_offset": tensor.offset if tensor.path else None,
            }
        )
        offset += tensor.size
    component_counts = {
        component: sum(tensor.key.startswith(prefix) for tensor in tensors)
        for component, prefix in COMPONENT_PREFIXES
    }
    component_counts["tokenizer"] = sum(
        tensor.key == TOKENIZER_KEY for tensor in tensors
    )
    return {
        "format_version": FORMAT_VERSION,
        "variant": args.variant,
        "source_repo": args.source_repo,
        "source_revision": args.source_revision,
        "source_lock_sha256": metadata["llada_image.source_lock_sha256"],
        "output": args.output.name,
        "output_header_size": len(prefix),
        "output_tensor_data_size": offset,
        "output_size": len(prefix) + offset,
        "aio_header_sha256": hashlib.sha256(prefix).hexdigest(),
        "tensor_count": len(tensors),
        "component_tensor_counts": component_counts,
        "tensors": tensor_entries,
    }


def write_plan(path: Path, plan: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"plan already exists: {path}; pass --overwrite to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(plan, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(partial, path)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


def convert_locked(args: argparse.Namespace) -> None:
    plan_only = getattr(args, "plan_only", False)
    if not plan_only and args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {args.output}; pass --overwrite to replace it"
        )
    lock, entries = read_lock(args)
    metadata_files = download_metadata_files(args, entries)
    config = build_config(metadata_files, args.variant)
    validate_variant(config, args.variant)
    tensors, weight_entries = collect_tensors(args, entries, metadata_files)
    metadata = make_metadata(config, args, canonical_json_sha256(lock), tensors)

    if plan_only:
        plan_path = getattr(args, "plan_output", None)
        if plan_path is None:
            plan_path = args.output.with_suffix(f"{args.output.suffix}.plan.json")
        plan = build_plan(args, tensors, metadata)
        write_plan(plan_path.resolve(), plan, args.overwrite)
        print(
            f"Planned {plan['tensor_count']} tensors and {plan['output_size']} "
            f"output bytes in {plan_path.resolve()}"
        )
        return

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
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.retries < 1:
        parser.error("--retries must be at least 1")
    return args


if __name__ == "__main__":
    convert(parse_args())
