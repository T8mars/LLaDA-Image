# LLaDA-Image → ComfyUI Core Roadmap

This document is the implementation contract for a native ComfyUI port of
LLaDA-Image. It deliberately targets ComfyUI Core rather than a long-lived
custom-node wrapper.

## Scope and non-negotiable decisions

- [ ] Ship one AIO `.safetensors` checkpoint per model variant (Base and Turbo).
- [ ] Load the AIO file through Core's existing `CheckpointLoaderSimple`, returning
  separately managed `MODEL`, `CLIP`, and `VAE` objects.
- [x] Do not combine the two variants in one file: Base and Turbo contain different
  denoiser and conditioning weights.
- [x] Implement the runtime in native PyTorch with `comfy.ops`, ModelPatcher/offload,
  and Core attention/RoPE helpers. Do not depend on Diffusers, Transformers,
  flash-attn, Triton-only kernels, or `trust_remote_code`.
- [x] Reuse Core's existing `latent_formats.Flux2` and Flux2 VAE support. Do not
  duplicate the VAE implementation.
- [x] Preserve all three upstream modes: text-to-image, VQ-conditioned generation,
  and single-image instruction editing.
- [x] Match the upstream prompt templates, attention masks, latent packing,
  timestep/sigma transforms, output sign, CFG layout, and Base/Turbo sampling rules.
- [x] Land the smallest model-specific node surface possible; standard Core nodes
  remain the default graph building blocks.
- [x] BF16 parity is the first merge target. FP8 is a follow-up only after its
  custom MoE quantization and incomplete packaging assumptions are verified.

"Single model" means one distributable checkpoint file, not one monolithic
`torch.nn.Module`. MODEL, CLIP, and VAE must remain distinct internally so Core can
patch, load, offload, and reuse them correctly.

## Pinned reference inputs

- [x] Upstream source inspected at commit
  `b4dfa9a3e50d90d6718975ca1fa1b0edbc90d512`.
- [x] ComfyUI reference inspected at commit
  `e80c1570b6b44a2557d5d8e341e05782d18c9bbb`.
- [x] Native port guide inspected at commit
  `3facd9bda7a9ec4af0a9dd8b99001fddc1d7eca4`.
- [x] Persist exact Hugging Face revisions and per-file hashes in conversion
  manifests for Base and Turbo.
- [ ] Obtain or generate licensed reference inputs and golden outputs for all modes.

## AIO checkpoint contract

The converter will produce this stable top-level layout:

```text
model.diffusion_model.*
text_encoders.llada2.*
text_encoders.queryformer.*
text_encoders.text_projection.*
text_encoders.sigvq.*
text_encoders.tokenizer_json
vae.*
```

Safetensors metadata will include a JSON `config` entry with the transformer,
text encoder, QueryFormer, text projection, SigVQ, VAE, variant, and sampling
configuration. It will also include source repository/revision, license,
conversion version, dtype, and recommended workflow settings.

Acceptance criteria:

- [x] Converter reads HF shards without keeping multiple full copies in RAM.
- [x] Every source tensor is mapped exactly once; duplicate and unknown keys fail.
- [x] Tensor shapes, dtypes, counts, hashes, and metadata are verified after write.
- [ ] The output loads with `CheckpointLoaderSimple` with no missing or unexpected
  model keys.
- [ ] Separate Base and Turbo conversion manifests are reproducible.

## Native runtime architecture

### Diffusion model

- [x] Add `comfy/ldm/llada_image/` with the transformer and sequence preparation.
- [x] Infer dimensions and layer counts from state-dict shapes where possible.
- [x] Read non-inferable official constants from checkpoint `config`; only retain a
  documented fallback for the exact official release configuration.
- [x] Use `comfy.ops` for every weighted module.
- [x] Use Core optimized attention and shared Flux RoPE primitives where compatible.
- [x] Preserve caption/image/SigVQ sequence ordering, position IDs, padding masks,
  noise masks, editing dual timesteps, and the upstream output negation.
- [x] Add model detection and a supported-model entry with `ModelType.FLOW`, BF16
  and FP32 support, `latent_formats.Flux2`, `vae.` and `text_encoders.` prefixes.
- [x] Add a `BaseModel` subclass that packages caption features, masks, semantic
  features, and source latents into Core conditioning objects.

### Conditioning stack

- [x] Add a native LLaDA2 MoE text encoder using Core-managed operations.
- [ ] Adapt expert banks to `comfy.ops.MoEExperts` and validate both numerical
  parity and memory behavior against the upstream eager/fused path.
- [x] Implement the exact prompt renderer and tokenizer from embedded
  `tokenizer_json` bytes.
- [x] Implement QueryFormer and its asymmetric text/query attention mask.
- [x] Implement the six-layer text projection stack.
- [x] Implement SigVQ image encoder, codebook lookup, and semantic projection.
- [x] Implement deterministic block-diffusion VQ-token generation with the exact
  token offsets, block length, unmasking steps, CFG, thresholds, and size frontend.
- [x] Return caption features as normal CLIP conditioning and model-specific values
  as named conditioning extras.

### VAE and latents

- [x] Reuse Core Flux2 VAE detection, BN normalization, 2x2 packing, and managed
  encode/decode paths.
- [x] Reuse `EmptyFlux2LatentImage` for the target latent.
- [x] Enforce multiples of 16 for text/VQ and multiples of 32 for editing.
- [x] Verify source-image preprocessing and resizing separately for SigVQ and VAE.

## Minimal node surface

- [x] Standard `CLIPTextEncode` handles ordinary positive/negative text encoding.
- [x] `LLaDAImageVQConditioning` adds generated SigVQ semantic features to positive
  conditioning only.
- [x] `LLaDAImageEditConditioning` encodes the source image once, attaches semantic
  features only to positive conditioning, and attaches source latents to both
  positive and negative conditioning.
- [x] `LLaDAImageScheduler` reads the checkpoint variant and emits exact Base or
  Turbo sigmas.
- [x] A narrowly scoped Turbo stochastic-flow `SAMPLER` implements the upstream
  denoised/noise interpolation with an explicit ComfyUI seed.
- [x] Reuse `CFGGuider`, `RandomNoise`, `SamplerCustomAdvanced`, `VAEDecode`, and
  `SaveImage`; do not add redundant loader, latent, guider, or decode nodes.
- [x] Implement new nodes using the current v3 node API.

## Sampling contract

Base:

- [x] Default 50 steps and CFG 5.0.
- [x] Generate the exact Kumaraswamy-like pre-shift schedule used upstream.
- [x] Use shift 1.0 and standard Euler with an explicit terminal zero.

Turbo:

- [x] Default 4 steps and CFG 1.0.
- [x] Start from a uniform pre-shift grid, apply rational flow shift 3.0, and append
  terminal zero.
- [x] Implement each stochastic step as
  `x0 = sample - sigma * model_output` followed by
  `next = (1 - sigma_next) * x0 + sigma_next * noise`.
- [x] Seed every per-step noise draw through the ComfyUI noise source. Document the
  determinism improvement over the upstream pipeline's global-RNG behavior.

## Parity and regression test matrix

- [x] State-dict detection, config parsing, prefix mapping, tokenizer extraction.
- [x] Exact tokenizer IDs/masks for empty, English, Chinese, long, and special-token
  prompts.
- [x] Component tests for QueryFormer, text projection, MoE routing, SigVQ codebook,
  VAE packing/BN, and diffusion transformer blocks.
- [x] Exact Base and Turbo sigma golden arrays.
- [x] Exact seeded Turbo single-step and multi-step sampler tests.
- [x] Positive/negative CFG tests: semantic features are positive-only; source
  latents are present in both branches.
- [ ] End-to-end text, VQ, and editing tests for Base and Turbo.
- [ ] 1024x1024, non-square, batch, dimension-validation, CPU construction,
  BF16 execution, low-VRAM offload/unload/reload, and NaN checks.
- [x] Compare captured intermediate tensors before judging final-image parity.
- [ ] Publish reference images with prompt, seed, resolution, steps, CFG, sampler,
  scheduler, checkpoint revision, and measurable error tolerances.

## Upstreaming sequence

Design proposal and maintainer questions were posted on the existing Core feature
issue: <https://github.com/Comfy-Org/ComfyUI/issues/16088#issuecomment-5540384427>.

- [x] Open a ComfyUI design/feature issue before the large implementation PR. Include
  the component map, AIO contract, node graph, sampling exception, licensing, model
  size, and parity strategy.
- [x] Request maintainer confirmation for AIO packaging, SigVQ ownership under CLIP,
  and the dedicated Turbo sampler.
- [ ] Keep the Core PR focused on native runtime, detection, minimal nodes, tests,
  and attribution.
- [ ] Publish conversion tooling/manifests and converted checkpoints separately from
  Core source if requested by maintainers.
- [ ] Add official workflow templates/model download metadata in the appropriate
  ComfyUI repository after Core support is accepted.
- [ ] Address review feedback without replacing the verified reference behavior with
  approximations.

## Completion gate

This roadmap is complete only when both BF16 variants load from one AIO file through
the standard checkpoint loader; all three modes run through native Core components;
Base and Turbo sampling match the pinned reference within declared tolerances; tests
cover model management and low-VRAM operation; reproducible workflows and conversion
manifests are published; and the Core contribution has been submitted with the
required evidence. A partial custom-node implementation does not satisfy this gate.
