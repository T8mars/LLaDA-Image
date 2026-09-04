#!/usr/bin/env python3
"""Verify official LLaDA-Image tensor shapes against the native ComfyUI port."""

from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import torch
from convert_comfyui_aio import COMPONENT_PREFIXES

VALIDATED_COMPONENTS = {
    "transformer",
    "text_encoder",
    "queryformer",
    "text_projection",
    "sigvq",
}


def remote_url(repo: str, revision: str, path: str) -> str:
    repo = urllib.parse.quote(repo, safe="/")
    revision = urllib.parse.quote(revision, safe="")
    path = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/{repo}/resolve/{revision}/{path}"


def read_url(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {"User-Agent": "llada-image-comfyui-contract-verifier/1"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        if byte_range is not None and response.status != 206:
            raise RuntimeError(f"server did not honor byte range for {url}")
        return response.read()


def read_remote_json(repo: str, revision: str, path: str) -> dict:
    value = json.loads(read_url(remote_url(repo, revision, path)))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def read_remote_safetensors_header(repo: str, revision: str, path: str) -> dict:
    url = remote_url(repo, revision, path)
    length_data = read_url(url, (0, 7))
    if len(length_data) != 8:
        raise ValueError(f"{path}: truncated safetensors length")
    header_length = struct.unpack("<Q", length_data)[0]
    if header_length < 2 or header_length % 8:
        raise ValueError(f"{path}: invalid safetensors header length {header_length}")
    header_data = read_url(url, (8, 7 + header_length))
    if len(header_data) != header_length:
        raise ValueError(f"{path}: truncated safetensors header")
    header = json.loads(header_data.decode("utf-8").rstrip(" "))
    if not isinstance(header, dict):
        raise TypeError(f"{path}: safetensors header must be an object")
    return header


def source_shapes(manifest: dict) -> dict[str, tuple[int, ...]]:
    repo = manifest["source_repo"]
    revision = manifest["source_revision"]
    prefixes = dict(COMPONENT_PREFIXES)
    shapes = {}
    for entry in manifest["files"]:
        path = entry["path"]
        component = path.split("/", 1)[0]
        if component not in VALIDATED_COMPONENTS or not path.endswith(".safetensors"):
            continue
        header = read_remote_safetensors_header(repo, revision, path)
        for key, tensor in header.items():
            if key == "__metadata__":
                continue
            destination = f"{prefixes[component]}{key}"
            if destination in shapes:
                raise ValueError(f"duplicate source tensor {destination}")
            shapes[destination] = tuple(tensor["shape"])
    return shapes


def module_shape_tensors(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    tensors = {
        name: torch.empty(parameter.shape, device="meta", dtype=parameter.dtype)
        for name, parameter in module.named_parameters()
    }
    tensors.update(
        {
            name: torch.empty(buffer.shape, device="meta", dtype=buffer.dtype)
            for name, buffer in module.named_buffers()
        }
    )
    for name, child in module.named_modules():
        if hasattr(child, "_orig_shape"):
            prefix = f"{name}." if name else ""
            tensors[f"{prefix}weight"] = torch.empty(
                child._orig_shape, device="meta", dtype=torch.bfloat16
            )
            bias = getattr(child, "bias", None)
            if bias is not None:
                tensors[f"{prefix}bias"] = torch.empty(
                    bias.shape, device="meta", dtype=bias.dtype
                )
    return tensors


def native_shapes(
    comfyui_root: Path, repo: str, revision: str, variant: str
) -> dict[str, tuple[int, ...]]:
    sys.path.insert(0, str(comfyui_root))
    from comfy.cli_args import args as comfy_args

    comfy_args.cpu = True

    import comfy.ops
    import comfy.supported_models
    from comfy.ldm.llada_image.model import LLaDAImage
    from comfy.text_encoders.llada_image import LLaDAImageTEModel

    transformer_config = read_remote_json(repo, revision, "transformer/config.json")
    text_config = read_remote_json(repo, revision, "text_encoder/config.json")
    queryformer_config = read_remote_json(repo, revision, "queryformer/config.json")
    projection_config = read_remote_json(repo, revision, "text_projection/config.json")
    sigvq_config = read_remote_json(repo, revision, "sigvq/config.json")

    transformer_keys = {
        "all_patch_size",
        "all_f_patch_size",
        "in_channels",
        "dim",
        "n_layers",
        "n_refiner_layers",
        "n_heads",
        "norm_eps",
        "qk_norm",
        "cap_feat_dim",
        "semantic_feat_dim",
        "rope_theta",
        "t_scale",
        "axes_dims",
    }
    with torch.device("meta"):
        diffusion_model = LLaDAImage(
            **{
                key: transformer_config[key]
                for key in transformer_keys
                if key in transformer_config
            },
            dtype=torch.bfloat16,
            device=torch.device("meta"),
            operations=comfy.ops.disable_weight_init,
        )
        clip_model = LLaDAImageTEModel(
            device=torch.device("meta"),
            dtype=torch.bfloat16,
            llada2_config=text_config,
            queryformer_config=queryformer_config,
            text_projection_config=projection_config,
            sigvq_config=sigvq_config,
        )

    expected = {
        f"model.diffusion_model.{key}": tuple(tensor.shape)
        for key, tensor in module_shape_tensors(diffusion_model).items()
    }
    adapter = comfy.supported_models.LLaDAImage(
        {"image_model": "llada_image", "variant": variant}
    )
    clip_checkpoint = adapter.process_clip_state_dict_for_saving(
        module_shape_tensors(clip_model)
    )
    expected.update({key: tuple(tensor.shape) for key, tensor in clip_checkpoint.items()})
    return expected


def verify(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    identity = {
        "variant": args.variant,
        "source_repo": args.source_repo,
        "source_revision": args.source_revision,
    }
    for key, expected in identity.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"{args.manifest}: expected {key}={expected!r}, got {manifest.get(key)!r}"
            )

    actual = source_shapes(manifest)
    expected = native_shapes(
        args.comfyui_root.resolve(),
        args.source_repo,
        args.source_revision,
        args.variant,
    )
    missing = sorted(set(actual) - set(expected))
    unexpected = sorted(set(expected) - set(actual))
    mismatched = sorted(
        key for key in actual.keys() & expected.keys() if actual[key] != expected[key]
    )
    if missing or unexpected or mismatched:
        details = [
            f"source-only keys: {missing[:10]}",
            f"native-only keys: {unexpected[:10]}",
            "shape mismatches: "
            + repr(
                [
                    (key, actual[key], expected[key])
                    for key in mismatched[:10]
                ]
            ),
        ]
        raise ValueError("LLaDA-Image shape contract failed; " + "; ".join(details))
    print(
        f"Verified {len(actual)} {args.variant} tensors against native ComfyUI "
        f"at {args.comfyui_root.resolve()}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", choices=("base", "turbo"), required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-revision", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    verify(parse_args())
