# Native ComfyUI packaging

This repository carries the reproducible converter for the native ComfyUI Core
implementation. The converter produces one AIO safetensors checkpoint per variant.
ComfyUI still exposes the contents as separate `MODEL`, `CLIP`, and `VAE` objects so
that its normal patching, loading, and offloading machinery remains in control.

## Pinned BF16 sources

| Variant | Hugging Face repository | Revision |
| --- | --- | --- |
| Base | `inclusionAI/LLaDA-Image` | `e4e2703f410f7ddb6ee8d6b09dac6a8ec5093039` |
| Turbo | `inclusionAI/LLaDA-Image-Turbo` | `f4afc52d925bbac4e22a1c947111fc1f127e37e5` |

The corresponding files, sizes, and SHA-256 values are locked in
`manifests/llada-image-base.source.json` and
`manifests/llada-image-turbo.source.json`. Conversion fails before writing if the
repository, revision, file set, size, or hash differs.

## Convert

The complete source repository is approximately 49 GB. The output has roughly the
same size and conversion needs additional temporary/free space for the output.

```bash
hf download inclusionAI/LLaDA-Image \
  --revision e4e2703f410f7ddb6ee8d6b09dac6a8ec5093039 \
  --local-dir models/LLaDA-Image-e4e270

python scripts/convert_comfyui_aio.py \
  models/LLaDA-Image-e4e270 \
  LLaDA-Image-Base-BF16-AIO.safetensors \
  --variant base \
  --source-repo inclusionAI/LLaDA-Image \
  --source-revision e4e2703f410f7ddb6ee8d6b09dac6a8ec5093039
```

For Turbo, use repository `inclusionAI/LLaDA-Image-Turbo`, revision
`f4afc52d925bbac4e22a1c947111fc1f127e37e5`, and `--variant turbo`.

The converter streams tensor payloads in 16 MiB chunks, validates sharded weight
indexes and official component key spaces, embeds the tokenizer and every component
configuration, atomically installs the output, rereads its structure and SHA-256,
and writes `<checkpoint>.manifest.json`.

## Core graph contract

- Load either AIO file with `CheckpointLoaderSimple`.
- Ordinary text-to-image uses two standard `CLIPTextEncode` nodes and
  `EmptyFlux2LatentImage`.
- VQ-conditioned generation replaces those text nodes with
  `LLaDAImageVQConditioning`; the target latent still comes from
  `EmptyFlux2LatentImage`.
- Editing uses `LLaDAImageEditConditioning`, which returns the positive and negative
  conditioning plus a target latent matching the encoded source image.
- `LLaDAImageScheduler` selects the exact Base or Turbo schedule from checkpoint
  metadata. Steps `0` means the official default: 50 for Base and 4 for Turbo.
- Base uses the standard Euler sampler with CFG 5.0. Turbo uses
  `SamplerLLaDAImageTurbo` with CFG 1.0 because the upstream scheduler injects fresh
  noise at every step. Its draws use ComfyUI's explicit seed rather than upstream's
  implicit global RNG, making repeated workflows deterministic.
- The remaining nodes are standard Core nodes: `RandomNoise`, `CFGGuider`,
  `SamplerCustomAdvanced`, `VAEDecode`, and `SaveImage`.

The implementation and evidence checklist are tracked in `roadmap.md`.
