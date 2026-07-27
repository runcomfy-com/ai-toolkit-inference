# Phase 0 research approach (from the Krea 2 rollout)

Getting the pipeline right depends entirely on understanding the arch first.
For Krea 2 the analysis was fanned out across parallel readers, one per
dimension, each required to quote file:line — then the risky claims were
adversarially re-checked before any code was written. That caught two things
a from-memory pass would have gotten wrong:

- **turbo's `mu` is NOT pinned to 1.15.** Krea's CLI passes `--mu 1.15`, but
  no ai-toolkit krea2 preset sets `model_kwargs.schedule_mu`, so training
  previews use the resolution-interpolated `mu`. Parity means following the
  trainer, not the vendor CLI.
- **the LoRA merge must not filter on a `blocks.` prefix** — the text tower is
  `txtfusion.{layerwise,refiner}_blocks`, which a prefix filter drops.

## The dimensions worth a dedicated reader

1. **How to add a model here** — derive the touchpoint checklist from real
   git history (`git log --diff-filter=A -- src/pipelines/`, `git show --stat`
   on the last few model-add commits), not from docs.
2. **How ai-toolkit-backed pipelines work** — the import shim, what
   `_load_pipeline` builds, which `BasePipeline` methods each overrides,
   `LoraMergeMethod.CUSTOM` end to end. chroma.py / flux2.py are the templates.
3. **The target arch itself** — components, forward/sampling contract, turbo
   vs raw, edit mode, LoRA key format. (Training repo
   `extensions_built_in/diffusion_models/<arch>/`.)
4. **Sampling parity** — how the trainer renders preview samples
   (`generate_images` / the arch's `get_generation_pipeline`), scheduler,
   seed→noise, and the concrete "things that cause drift".
5. **Peripheral surface** — download_config, request/response schemas, the
   executor, the ComfyUI nodes, the tests that enumerate models, the docs
   templates.
6. **LoRA format parity** — what ai-toolkit writes to disk for this arch and
   how existing pipelines load it.
7. **External facts** (needs web) — do the HF repos exist / are they gated,
   is there a native diffusers pipeline, recommended settings, licence.
8. **Runtime & deploy** — version alignment, dependency pins (does the arch
   need a transformers/diffusers bump?), repo conventions, is the repo public.

## The rule that mattered most

Every load-bearing claim gets verified against the actual file before it goes
into code. "Adversarially verify your findings." The dossier was useful, but
the things that would have caused silent wrong-output were all caught in the
verify pass, not the research pass.

The Krea 2 spec that came out of this lives (for reference) in the session
scratchpad; the durable distillation is the parent SKILL.md.
