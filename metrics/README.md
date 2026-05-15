# `metrics/`

Region-aware evaluation scripts and aggregated results for every
experiment reported in the paper. The two scoring functions
(LPIPS-based and CLIP-based) are defined in §5.1; this directory hosts
their implementations and the per-concept CSV / JSON outputs.

## Layout

```
metrics/
├── eval_masked.py             # LPIPS_LOC / CLIP_LOC for the external mask-guided experiments
├── eval_selectivity.py        # LPIPS_SEL / CLIP_SEL for the subject-anchored experiments
├── summarize_masked.py        # per-concept LPIPS_LOC / CLIP_LOC medians
├── summarize_selectivity.py   # per-concept LPIPS_SEL / CLIP_SEL medians
├── select_best_runs.py        # pick top-k qualitative samples per concept
├── clip_score.py              # CLIP scoring helpers, used internally
├── lpip_score.py              # LPIPS scoring helpers, used internally
├── generate_images_*.py       # baseline / reference generation scripts inherited
│                              # from the upstream Concept Sliders codebase
├── results_sdxl_masked/       # per-concept CSV + aggregate JSON (paper tab:appendix:results-masked-sdxl)
├── results_sdxl_selectivity/  # per-concept CSV + aggregate JSON (paper tab:appendix:results-anchored)
└── results_flux_masked/       # per-concept CSV + aggregate JSON (paper tab:appendix:results-masked-flux)
```

## Origin

The two core evaluation scripts (`eval_masked.py`,
`eval_selectivity.py`) and their summarisers
(`summarize_masked.py`, `summarize_selectivity.py`) are original to
this work. They implement the two general scores defined in §5.1 of the
paper:

```
LPIPS_AB = d_a / (d_b + eps)              with  d_x = LPIPS(I0*m_x, I1*m_x) / (A_x + eps)
CLIP_AB  = delta_a / (delta_a + |delta_b| + eps)   if  delta_a > 0,  else 0
                                                    with  delta_x = C(I1 * m_x) - C(I0 * m_x)
```

instantiated with two different region pairs:

- ``(m_a, m_b) = (m, m_complement)`` for the **localization** metrics
  ``LPIPS_LOC`` / ``CLIP_LOC`` used in §4.2.
- ``(m_a, m_b) = (m_target, m_nontarget)`` for the **selectivity**
  metrics ``LPIPS_SEL`` / ``CLIP_SEL`` used in §4.1.

The `generate_images_*.py` scripts are inherited from the upstream
Concept Sliders codebase
([rohitgandikota/sliders](https://github.com/rohitgandikota/sliders),
MIT licensed); they are not used by the paper experiments but are
kept here for compatibility with downstream comparisons.

## Usage

All commands are launched from the repository root. The scripts auto-
detect the slider scales present in each run directory; per-run JSON
results are written next to the edited images and aggregated CSV /
JSON live under the corresponding `results_<arch>_<task>/<concept>/`
folder.

### External mask-guided (paper §4.2)

```bash
# Run for one concept:
python metrics/eval_masked.py --concept smile_person --device cuda
# Aggregate medians across the 20 runs of each concept:
python metrics/summarize_masked.py
```

Inputs (per run dir): `base.png`, `edited_<concept>_s{1,2,3}.png`,
`mask_target.png`.
Outputs: `metrics/results_<arch>_masked/<concept>/eval_results.csv`
and `eval_aggregate.json`.

### Subject-Anchored Training (paper §4.1)

```bash
# Compute the selectivity metrics for one concept:
python metrics/eval_selectivity.py --concept age --device cuda
# Aggregate medians across the 20 runs of each concept:
python metrics/summarize_selectivity.py
```

Inputs (per run dir): `base.png`,
`edited_<concept>_specific_s{0.5,1.0,1.5}.png`,
`edited_<concept>_general_s{0.5,1.0,1.5}.png`,
`mask_target.png`, `mask_nontarget.png`.
Outputs: `metrics/results_sdxl_selectivity/<concept>/eval_results.csv`
and `eval_aggregate.json`.

### Picking qualitative samples

For the qualitative figures in the paper, `select_best_runs.py` ranks
the runs of each concept by their localization / selectivity score and
prints the top-k filenames:

```bash
python metrics/select_best_runs.py \
       --results_dir metrics/results_sdxl_masked \
       --runs_root   sdxl/tasks/masked_lora/runs \
       --top_k       3
```

## Implementation notes

- **LPIPS backbone**: AlexNet (`lpips.LPIPS(net="alex")`), matching the
  convention of the paper. The image tensors are mapped to ``[-1, 1]``
  before being multiplied by the mask, which is the way the masked
  edit operator was applied at inference too — so what we measure is
  the LPIPS of the masked composite vs the masked composite of the
  base image.
- **CLIP backbone**: OpenAI `clip-vit-base-patch32`. Inside / outside
  the mask is composited against uniform mid-grey (RGB 127) so that
  CLIP evaluates the masked region in a plausible full-image context,
  rather than being tricked by a black background.
- **Aggregation**: medians over the 20 runs of each concept, computed
  by `summarize_*.py`. The choice of median over mean is justified in
  §5.1 of the paper (LPIPS scores are unbounded ratios and the CLIP
  scores often distribute bimodally between successful and failed
  edits).
- **eps**: `1e-8` everywhere, both in the LPIPS denominator and in the
  CLIP normalisation.

## Reproducibility

The exact tables reported in the paper appendix
(`tab:appendix:results-masked-sdxl`, `tab:appendix:results-masked-flux`,
`tab:appendix:results-anchored`) can be regenerated by running the
corresponding `summarize_*.py` script over the committed
`results_*/<concept>/eval_results.csv` files. The CSVs themselves were
produced by `eval_*.py` after each evaluation pipeline (the SLURM
templates that orchestrate the full chain live under each task's
`jobs/new_slurm/` folder).
