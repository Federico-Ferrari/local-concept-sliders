# `sdxl/tasks/selectivity/`

Evaluation pipeline for the **Subject-Anchored Training** mechanism
described in §4.1 of the paper. For each anchored slider trained under
[`sdxl/trained_sliders/training/`](../../trained_sliders/training/),
this task generates a set of multi-subject scenes, applies the slider
**globally** (no spatial mask, no inference-time gating) and computes
the region-aware selectivity metrics on the resulting images. The
hypothesis under test is whether a subject-anchored slider concentrates
its effect on the intended subject more than its un-anchored
counterpart.

The selectivity metrics themselves are not produced here — they are
computed afterwards by `metrics/eval_selectivity.py` using SAM masks
``mask_target.png`` and ``mask_nontarget.png`` (see §5.1 of the paper).

## Pipeline

```
Phase 1 (HPC)              Phase 2 (local, interactive)   Phase 3 (HPC)
generate 20 base scenes -> SAM target + non-target mask -> apply slider GLOBALLY
                                                          (specific + general,
                                                           3 scales each)
```

- **Phase 1** generates 20 multi-subject scenes per concept (fixed
  seeds), reusing [`sdxl/tasks/masked_lora/scripts/01_generate_base.py`](../masked_lora/scripts/01_generate_base.py)
  via the SLURM templates. The base image, the initial latents and the
  metadata are saved in `runs/eval_<concept>_<seed>/`.
- **Phase 2** segments two SAM masks per scene — one on the target
  subject and one on the non-target subject — using
  [`mask_SAM/choose_masks_dual.py`](../../../mask_SAM/choose_masks_dual.py).
- **Phase 3** runs `scripts/03_apply_slider.py` to apply each slider
  globally on the same scene and seed. For every concept this task
  produces both the specific (subject-anchored) and the general
  (un-anchored) edits at 3 scales each, sharing the same base images so
  the comparison is paired per scene.

Phase 3 is the only one that calls the local script
`scripts/03_apply_slider.py`; phase 1 reuses the masked_lora phase-1
script and phase 2 reuses the shared SAM tools. The masks produced in
phase 2 are not used at inference (the slider is applied globally) —
they only enter the metrics computation.

## Origin

The whole pipeline is original to this work. The slider stack
(`LoRANetwork`, `apply_to`, `set_lora_slider`) lives under
[`sdxl/core/`](../../core/) and is adapted from
[rohitgandikota/sliders](https://github.com/rohitgandikota/sliders);
this task only uses the public surface of that library.

## Layout

```
sdxl/tasks/selectivity/
├── scripts/
│   └── 03_apply_slider.py     # Phase 3: global slider application, multi-scale
├── jobs/
│   └── new_slurm/             # SLURM templates used for the paper
└── runs/                      # per-evaluation run directories (git-ignored)
```

## Usage

All commands are launched from the repository root. The four concepts
evaluated in the paper (`tab:appendix:results-anchored`) are
``age``, ``curlyhair``, ``furlength``, ``smile``.

### Phase 1: 20 base scenes per concept

```bash
sbatch sdxl/tasks/selectivity/jobs/new_slurm/eval_age_phase1.slurm
# also: eval_curlyhair_phase1.slurm, eval_furlength_phase1.slurm, eval_smile_phase1.slurm
```

Each script generates the 20 multi-subject scenes per concept with
fixed seeds. The exact prompts and seeds are committed in this
repository under `jobs/new_slurm/` (rather than in a paper appendix);
the outputs land in `runs/eval_<concept>_<seed>/`.

### Phase 2: SAM target + non-target masks (local)

```bash
python mask_SAM/choose_masks_dual.py \
       --runs_root sdxl/tasks/selectivity/runs \
       --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
       --run_ids $(ls sdxl/tasks/selectivity/runs/ | grep '^eval_')
```

For each run the script asks for two clicks (target subject and
non-target subject), saves `mask_target.png` and `mask_nontarget.png`
inside the run directory.

### Phase 3: apply specific + general slider at 3 scales

```bash
sbatch sdxl/tasks/selectivity/jobs/new_slurm/eval_age_phase3.slurm
# (same for curlyhair, furlength, smile)
```

Each phase-3 SLURM iterates the 20 runs of the concept and calls
`scripts/03_apply_slider.py` twice — once with the subject-anchored
slider (``--output_prefix edited_<concept>_specific``) and once with the
un-anchored counterpart (``--output_prefix edited_<concept>_general``)
— at three scales each. The slider checkpoints to use are configured at
the top of the SLURM file.

### Metrics

After phases 1–3 are complete, compute the selectivity metrics:

```bash
sbatch sdxl/tasks/selectivity/jobs/new_slurm/run_eval_selectivity.slurm
# or, for all 4 concepts:
sbatch sdxl/tasks/selectivity/jobs/new_slurm/run_eval_selectivity_all.slurm
```

These delegate to `metrics/eval_selectivity.py`, which loads each
`mask_target.png` / `mask_nontarget.png` pair and computes
``LPIPS_SEL`` and ``CLIP_SEL`` (paper §5.1). Aggregated results land
under `metrics/results_sdxl_selectivity/<concept>/`.

## Useful parameters

`scripts/03_apply_slider.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--slider_scales` | `[1.0]` | One or more slider scales to sweep (the model is loaded once and reused across scales). |
| `--start_noise` | 750 | Timestep threshold above which LoRA is not applied. Same convention as the masked_lora pipeline. |
| `--rank` | 4 | LoRA rank of `LoRANetwork`. |
| `--output_prefix` | required | Prefix for the output PNGs and per-edit metadata files (e.g. `edited_age_specific`). |
| `--dtype` | float16 | UNet dtype. |

## Notes

- The slider is applied **globally** in this task (no spatial mask at
  inference time). The masks are produced only so that the
  region-aware metrics can compare the change inside the target vs the
  non-target subject afterwards.
- Phase 1 reuses the base-generation script from `sdxl/tasks/masked_lora/`
  so the trajectory is identical to the one used by the mask-guided
  evaluation, which means the base images are pixel-identical when the
  two tasks share a seed (useful for visual side-by-side comparisons).
- The paired structure (specific vs general slider on the same scene
  and seed) is what enables the per-scene paired comparison reported
  in the paper. Run dirs that are missing one of the two edit sets
  should be regenerated before the metrics step.
