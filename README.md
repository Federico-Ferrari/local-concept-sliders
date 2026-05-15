# Where to Slide — Code Release

Reference implementation accompanying the work for the course **Computer Vision
and Image Processing**:
**"Where to Slide: Localizing Concept Sliders for Multi-Subject Image Editing"**
(F. Ferrari, T. Errico, E. Paccagnella, R. Rinero — Bocconi University).

The repository implements the two localization mechanisms proposed in the paper —
**subject-anchored training** (§4.1) and **external mask-guided application** (§4.2) —
on two text-to-image backbones (SDXL and Flux.1-dev), together with the
LoRAShop-based internal-mask alternative discussed in §4.2 and the real-image
verification via Tight Inversion mentioned in §1. Localization is evaluated with
the region-aware LPIPS and CLIP metrics defined in §5.1.

> 📦 **Note.** The hand-drawn SAM masks and the full set of generated
> evaluation images are **not committed to this repository** because of
> their size. The **quantitative results** derived from them are
> committed under [`metrics/results_*/`](metrics/), and the
> **qualitative figures** used in the paper (selected from the
> evaluation set) are committed under
> [`paper_figures/`](paper_figures/). All the **prompts and seeds**
> used in the evaluation are present in the repository. If the raw
> evaluation images / masks are needed, please contact any of the
> authors of the paper and we will provide them directly.

---

## Table of contents

1. [Repository layout](#repository-layout)
2. [Setup](#setup)
3. [Reproducing the experiments](#reproducing-the-experiments)
   - [Subject-Anchored Training (§4.1)](#1-subject-anchored-training-41)
   - [External Mask-Guided Application (§4.2)](#2-external-mask-guided-application-42)
   - [LoRAShop adaptation, internal-mask alternative (§4.2)](#3-lorashop-adaptation-internal-mask-alternative-42)
   - [Real-image verification via Tight Inversion (§1)](#4-real-image-verification-via-tight-inversion-1)
4. [Evaluation and results folder](#evaluation-and-results-folder)
5. [HPC / SLURM templates](#hpc--slurm-templates)
6. [Acknowledgments](#acknowledgments)
7. [License](#license)

---

## Repository layout

The repository mirrors the two backbones under `sdxl/` and `flux/`, with a shared
mask-extraction stage under `mask_SAM/` and a shared evaluation stage under `metrics/`.

```
local-concept-sliders/
├── sdxl/                                # SDXL base 1.0 stack
│   ├── core/                            # Concept-Sliders LoRA core (adapted)
│   │   ├── lora.py                      # LoRANetwork / LoRAModule
│   │   ├── train_util.py                # diffusion + CFG helpers
│   │   ├── model_util.py                # model loading
│   │   ├── prompt_util.py               # prompt-pair training objective
│   │   └── config_util.py               # YAML config schema
│   ├── trained_sliders/
│   │   ├── training/                    # slider training entrypoint
│   │   │   ├── scripts/                 # train.py, train_with_preservation.py
│   │   │   ├── configs/                 # one YAML per slider (paper §5.2)
│   │   │   └── prompts/                 # target / positive / unconditional triples
│   │   └── sliders/                     # trained .pt outputs (git-ignored)
│   └── tasks/
│       ├── baseline/                    # global slider application, dose-response
│       ├── masked_lora/                 # §4.2 external mask-guided pipeline
│       ├── selectivity/                 # §4.1 evaluation pipeline (multi-subject)
│       ├── real_editing/                # §1 Tight-Inversion-based real editing
│       └── masked_lora_editing/         # earlier SD1.4 real-edit prototype (superseded)
│
├── flux/                                # Flux.1-dev stack, mirrors sdxl/
│   ├── core/                            # LoRA core ported to Flux MM-DiT attention
│   ├── trained_sliders/{training,sliders}/
│   └── tasks/
│       ├── baseline/                    # global slider application
│       ├── masked_lora/                 # §4.2 external mask-guided pipeline (Flux)
│       └── shop_concept/                # §4.2 LoRAShop-based internal-mask variant
│
├── mask_SAM/                            # SAM (ViT-H) wrappers used at Phase 2
│   ├── segment_with_sam.py              # single-image SAM CLI
│   ├── choose_masks.py                  # batch interactive masking (target only)
│   ├── choose_masks_dual.py             # batch interactive (target + non-target)
│   ├── place_masks.py                   # copy masks into the per-run directories
│   └── checkpoints/                     # SAM weights, downloaded on first use
│
├── metrics/                             # paper §5 metrics and aggregations
│   ├── eval_masked.py                   # LPIPS_LOC / CLIP_LOC (§4.2 setting)
│   ├── eval_selectivity.py              # LPIPS_SEL / CLIP_SEL (§4.1 setting)
│   ├── summarize_masked.py              # per-concept LPIPS_LOC / CLIP_LOC medians
│   ├── summarize_selectivity.py         # per-concept LPIPS_SEL / CLIP_SEL medians
│   ├── select_best_runs.py              # pick top-k qualitative samples
│   └── results_*/                       # CSV + aggregate JSON per concept
│
├── tools/                               # optional HPC sync utilities (templates)
│   ├── set_slurms.sh.example            # per-user HPC environment template
│   ├── pull_config.sh.example           # per-user local environment template
│   ├── push_to_hpc.sh                   # rsync local → HPC (slurm + configs)
│   └── pull_from_hpc.sh                 # rsync HPC → local (sliders + outputs)
│
└── README.md
```

Each `<arch>/tasks/<task>/` follows the same convention:

```
<task>/
├── scripts/      # Python entrypoints
├── jobs/new_slurm/  # SLURM templates used for the paper experiments
├── logs/         # SLURM stdout/stderr (git-ignored except .gitkeep)
├── outputs/      # per-run artefacts (populated locally, .gitkeep placeholder)
└── README.md     # task-level documentation
```

All Python imports are absolute from the repository root (e.g.
`from sdxl.core.lora import LoRANetwork`); every entrypoint must be launched
with the repository root on `PYTHONPATH` (the bundled SLURM templates do this
via `cd $SLURM_SUBMIT_DIR`).

---

## Setup

### Requirements

The two backbones use different libraries and are best installed in separate
environments.

| Backbone | Lockfile | Notes |
|---|---|---|
| SDXL | `sdxl/trained_sliders/training/requirements-sdxl.lock` | PyTorch + diffusers; CUDA 12.1 wheels |
| Flux.1-dev | `flux/trained_sliders/training/requirements-flux.lock` | PyTorch + diffusers + PEFT; tested with `torch==2.4.1+cu124` |

Reference installation scripts are provided as starting points:

```bash
# SDXL: standard pip install from the lockfile
python -m venv .venv-sdxl && source .venv-sdxl/bin/activate
pip install -r sdxl/trained_sliders/training/requirements-sdxl.lock

# Flux: lockfile or the bundled venv/conda setup scripts
python -m venv .venv-flux && source .venv-flux/bin/activate
pip install -r flux/trained_sliders/training/requirements-flux.lock
# alternative: ./flux/trained_sliders/training/setup_flux_venv.sh
# alternative: ./flux/trained_sliders/training/setup_sliders_flux_conda.sh
```

The local SAM stage has a separate, lightweight environment:

```bash
python -m venv .venv-sam && source .venv-sam/bin/activate
pip install torch torchvision numpy pillow matplotlib
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Pretrained checkpoints

The following weights are downloaded on first use (HuggingFace cache) or must be
placed under the indicated path. None of these are committed to the repository.

| Checkpoint | Used by | Source |
|---|---|---|
| `stabilityai/stable-diffusion-xl-base-1.0` | SDXL backbone | HuggingFace |
| `black-forest-labs/FLUX.1-dev` | Flux backbone | HuggingFace (gated) |
| `h94/IP-Adapter` (`ip-adapter-plus_sdxl_vit-h.safetensors` + ViT-H image encoder) | Tight Inversion (§1) | HuggingFace |
| `sam_vit_h_4b8939.pth` (~2.4 GB) | SAM Phase 2 | `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth` → `mask_SAM/checkpoints/` |

Trained slider weights produced by the experiments live under
`<arch>/trained_sliders/sliders/`. The final sliders used by the paper
runs are checked in under `general/` (un-anchored) and `anchored/`
(subject-anchored) via a whitelist in `.gitignore`; any other slider
checkpoints written under that directory are git-ignored and regenerated
by re-running training (see §[Subject-Anchored Training](#1-subject-anchored-training-41)).

---

## Reproducing the experiments

The four code paths below cover all experiments reported in the paper. Every
entrypoint accepts CLI flags; the bundled SLURM templates in `<task>/jobs/`
are reference invocations and can be adapted to any GPU host. Run all commands
from the repository root.

### 1. Subject-Anchored Training (§4.1)

A subject-anchored slider is trained with the standard Concept-Sliders prompt
recipe restricted to a specific subject class, optionally augmented with
counter-anchor entries (paper Fig. 1). Configurations live in
`<arch>/trained_sliders/training/configs/` (one YAML per slider) and prompt
triples in `<arch>/trained_sliders/training/prompts/`.

```bash
# SDXL — train one anchored slider (here, age on woman)
python sdxl/trained_sliders/training/scripts/train.py \
       --config_file sdxl/trained_sliders/training/configs/age_woman_sdxl_v1.yaml

# Flux — same idea, different entrypoint
python flux/trained_sliders/training/scripts/train_flux_slider.py \
       --config_file flux/trained_sliders/training/configs/<your_config>.yaml
```

Trained sliders are written to `<arch>/trained_sliders/sliders/<name>/<weights>`
(see `general/` and `anchored/` for the final paper checkpoints committed to
the repository).

To evaluate the selectivity of a trained slider against its un-anchored counterpart
on the 20 multi-subject scenes used in the paper:

```bash
# Phase 1: generate the 20 multi-subject scenes + apply both sliders
sbatch sdxl/tasks/selectivity/jobs/new_slurm/eval_age_phase1.slurm    # or run the script directly
sbatch sdxl/tasks/selectivity/jobs/new_slurm/eval_age_phase3.slurm

# Phase 2: produce target / non-target masks with SAM (Mac/local, interactive)
python mask_SAM/choose_masks_dual.py \
       --runs_root sdxl/tasks/selectivity/runs \
       --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
       --run_ids $(ls sdxl/tasks/selectivity/runs/)

# Phase 3: compute LPIPS_SEL and CLIP_SEL
python metrics/eval_selectivity.py --concept age --device cuda
python metrics/summarize_selectivity.py
```

Aggregated tables (matching `tab:appendix:results-anchored` in the paper) end up
in `metrics/results_sdxl_selectivity/<concept>/`.

### 2. External Mask-Guided Application (§4.2)

The mask-guided pipeline consists of three stages: generate a base scene,
segment it externally with SAM, and re-generate with the slider gated by the
mask using the per-step blend in eq. (2) of the paper. Each stage has a Python
entrypoint and matching SLURM templates.

```bash
# Phase 1 — base generation (saves base.png + metadata.json + init_latents)
python sdxl/tasks/masked_lora/scripts/01_generate_base.py \
       --prompt "a woman and a man sitting at a Parisian cafe" \
       --seed 1001 --run_id eval_age_person_01

python flux/tasks/masked_lora/scripts/01_generate_base.py \
       --prompt "<your prompt>" --seed 1001 --run_id <run>

# Phase 2 — interactive SAM masking on the base image (Mac/local)
python mask_SAM/choose_masks.py \
       --runs_root sdxl/tasks/masked_lora/runs \
       --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
       --run_ids $(ls sdxl/tasks/masked_lora/runs/ | grep '^eval_')

# Phase 3 — masked LoRA edit (single slider, multi-scale sweep)
python sdxl/tasks/masked_lora/scripts/03_masked_edit.py \
       --run_dir sdxl/tasks/masked_lora/runs/eval_age_person_01 \
       --slider_path sdxl/trained_sliders/sliders/general/age_person/age_person.safetensors \
       --slider_scale 2.0 --start_noise 700
```

Phase 3 supports the two extensions described in §4.2:

- **Multiple disjoint masks**, one slider each — pass `--mask_names m1.png m2.png`,
  `--slider_paths s1.pt s2.pt`, `--slider_to_mask 0 1`, `--slider_scales 2.0 2.0`.
- **Multiple sliders inside the same mask** — additive aggregation of LoRA updates,
  same flags with repeated mask indices in `--slider_to_mask` (e.g. `0 0 1`).

The Flux counterpart is `flux/tasks/masked_lora/scripts/03_masked_edit.py` with
the same CLI; the only backbone-specific detail (mask packing into Flux's
token grid) is handled internally — see §3.1 of the paper.

To compute the region-aware metrics on all 20 scenes of a concept:

```bash
python metrics/eval_masked.py --concept age_person --device cuda
python metrics/summarize_masked.py
```

Outputs land under `metrics/results_sdxl_masked/<concept>/`.

### 3. LoRAShop adaptation, internal-mask alternative (§4.2)

The internal-mask alternative we explored is a Flux-only pipeline that derives
masks from the model's own cross-attention rather than from an external
segmentation. It is bundled in `flux/tasks/shop_concept/` and described in
[`flux/tasks/shop_concept/README.md`](flux/tasks/shop_concept/README.md).
Our modifications on top of LoRAShop — per-slider continuous scales (the
paper's `s_i`) and additive composition of multiple sliders inside the
same region — are documented in the "Origin" section of that README.

Minimal run:

```bash
python flux/tasks/shop_concept/scripts/generate.py \
       --prompt "a man standing in front of a landscape" \
       --target_prompt "landscape" "man" \
       --slider_paths flux/trained_sliders/sliders/general/painterly/slider_0.pt \
                      flux/trained_sliders/sliders/general/age/slider_0.pt \
       --lora_scales 1.0 1.0 \
       --output_path flux/tasks/shop_concept/outputs/demo.png \
       --seed 42
```

This pipeline is the source of the qualitative example shown in
`fig:appendix:internal-masks` of the paper.

### 4. Real-image verification via Tight Inversion (§1)

`sdxl/tasks/real_editing/` is a self-contained three-stage pipeline used to
verify that the masked editing operator transfers from generated to real images.
The inversion backend is Tight Inversion (Kadosh et al., 2025) with IP-Adapter
conditioning; the masked-edit stage reuses the same per-step blend of §4.2.

```bash
# Stage 1: invert a real image (SDXL + Tight Inversion + IP-Adapter)
python sdxl/tasks/real_editing/scripts/invert_real_image.py \
       --image path/to/photo.jpg \
       --prompt "a portrait of a person" \
       --run_id real_001

# Stage 2: SAM mask on reconstruction.png (NOT on original.png)
python mask_SAM/segment_with_sam.py \
       --run_dir sdxl/tasks/real_editing/outputs/real_001 \
       --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
       --image_name reconstruction.png --mode interactive

# Stage 3: masked edit on the inverted latent
python sdxl/tasks/real_editing/scripts/edit_real_image_masked.py \
       --run_dir sdxl/tasks/real_editing/outputs/real_001 \
       --slider_path sdxl/trained_sliders/sliders/anchored/smile_woman/smile_woman.safetensors \
       --slider_scale 2.0
```

The earlier prototype at `sdxl/tasks/masked_lora_editing/` (real-image editing
on SD-1.4 and SDXL via DDIM/null-text inversion) is preserved for reference but
is superseded by `real_editing/`; it is not used in the paper.

---

## Evaluation and results folder

All quantitative results reported in the paper are reproduced by the scripts in
`metrics/`, and aggregated outputs are committed under `metrics/results_*/`:

| Folder | Paper table | Content |
|---|---|---|
| `metrics/results_sdxl_selectivity/` | `tab:appendix:results-anchored` | Subject-Anchored Training, SDXL |
| `metrics/results_sdxl_masked/` | `tab:appendix:results-masked-sdxl` | External Mask-Guided, SDXL |
| `metrics/results_flux_masked/` | `tab:appendix:results-masked-flux` | External Mask-Guided, Flux |

Each subdirectory contains `eval_results.csv` (one row per scene × scale)
and `eval_aggregate.json` (mean / std per scale). The internal-mask
alternative (LoRAShop adaptation, Appendix F of the paper) was not
quantitatively evaluated; its qualitative output is committed at
[`paper_figures/lorashop.png`](paper_figures/lorashop.png) and is
produced by the pipeline under
[`flux/tasks/shop_concept/`](flux/tasks/shop_concept/).

`metrics/select_best_runs.py` picks the top-k qualitative samples per concept
for inclusion in the paper figures (committed under `paper_figures/`).

---

## HPC / SLURM templates

Every task under `<arch>/tasks/<task>/jobs/new_slurm/` contains the SLURM
templates that were used to produce the paper's results on a Slurm-managed
GPU cluster. The templates are self-documenting CLI invocations of the
Python entrypoints and can be run on any GPU host by replacing the `sbatch`
call with `bash`.

The optional helper scripts in `tools/` (`push_to_hpc.sh`, `pull_from_hpc.sh`)
synchronize SLURM scripts and result artefacts between a local workstation and
a remote HPC node via `rsync`; their configuration files
(`tools/set_slurms.sh`, `tools/pull_config.sh`) are user-specific and
git-ignored. A user instantiates them from the bundled `.example` templates.
See `tools/README.md` for details. They are not required to run any experiment.

---

## Acknowledgments

This project builds on three publicly available prior works, whose code we
used as a starting point and adapted to our setting.

The whole repository starts from **Concept Sliders** (Gandikota et al., 2023,
[code](https://github.com/rohitgandikota/sliders)). We reused their LoRA
training framework and the prompt-based recipe that turns a LoRA adapter into
a continuous attribute knob, and built the two localization mechanisms of the
paper — subject-anchored training and external mask-guided application — on
top of it. The `core/` modules under `sdxl/` and `flux/`, together with the
training entrypoints under `<arch>/trained_sliders/training/scripts/`, are
direct adaptations of theirs.

The internal-mask alternative discussed in §4.2 of the paper, implemented in
`flux/tasks/shop_concept/`, adapts the **LoRAShop** pipeline (Dalva et al.,
2025, [code](https://github.com/gemlab-vt/LoRAShop)) for Flux.1-dev. We took
their multi-LoRA, attention-derived mask machinery and extended it to accept
Concept Slider checkpoints with continuous per-slider scales, and to compose
multiple sliders additively inside the same region.

The real-image verification mentioned in §1 of the paper, implemented in
`sdxl/tasks/real_editing/`, uses **Tight Inversion** as inversion backend
(Kadosh et al., 2025,
[HuggingFace Space](https://huggingface.co/spaces/tight-inversion/tight-inversion)).
We integrated their DDIM + gradient-descent + IP-Adapter inversion with the
same masked-edit operator used elsewhere in the repository.

We additionally rely on standard libraries (PyTorch, `diffusers`, PEFT) and
on [Segment Anything](https://github.com/facebookresearch/segment-anything)
(Kirillov et al., 2023) for the SAM mask extraction stage.

Everything else in the repository — in particular the masked editing operator
and its multi-mask / multi-LoRA extension, the region-aware LPIPS and CLIP
metrics of §5.1, the subject-anchored training configurations and prompt
triples of §4.1, the evaluation pipelines under `metrics/`, and the local
utilities under `mask_SAM/` and `tools/` — is original to this work.

---

## License

The original code in this repository is released under the **MIT License**.
Files adapted from upstream projects retain their original licenses, which are
compatible with MIT (MIT for Concept Sliders and LoRAShop, Apache-2.0 for
diffusers and Segment Anything). The Tight-Inversion HF Space does not ship an
explicit license file; we treat the adapted excerpts as research-only reference
and credit the authors accordingly.
