# `sdxl/trained_sliders/training/`

Training framework for SDXL Concept Sliders. The sliders trained here are
the inputs to every SDXL evaluation in the paper:

- The subject-anchored sliders trained with the prompt recipe of §4.1
  (e.g. `age_woman_sdxl_v1`, `smile_woman_sdxl_v1`, `curlyhair_man_sdxl_v1`,
  `furlength_dog_sdxl_v1`) and their un-anchored counterparts
  (`age_person`, `smile_person`, `curlyhair_person`, `furlength_pet`).
- Two style/scene sliders used by the mask-guided experiments of §4.2
  (`daynight_sdxl_v1`, `painterly_sdxl_v1`).

## Origin

The training script is adapted from the upstream Concept Sliders
codebase ([rohitgandikota/sliders](https://github.com/rohitgandikota/sliders),
MIT licensed). The only structural changes are:

- the LoRA stack lives in [`sdxl/core/`](../../core/) (so it can be
  shared by the editing tasks downstream);
- a `sys.path` insertion at the top of each training script;
- `scripts/train_with_preservation.py` adds an explicit preservation
  term on a second prompt set — an alternative way of obtaining the
  same effect as the YAML-trick implementation of §4.1 in the paper.
  Both routes are implemented; the runs reported in the paper use the
  YAML-trick variant on top of the upstream `train.py`.

Configurations (`configs/`) and prompt sets (`prompts/`) are
original to this work.

## Layout

```
sdxl/trained_sliders/training/
├── scripts/
│   ├── train.py                        # upstream Concept Sliders training
│   └── train_with_preservation.py      # variant with an explicit preservation loss
├── configs/                            # one YAML per slider (paper §5.2)
├── prompts/
│   ├── new_prompt/                     # prompt sets used for the paper runs
│   └── old_prompt/                     # earlier prompt iterations + upstream copies
├── jobs/
│   └── new_slurm/                      # SLURM templates used for the paper
├── requirements-sdxl.lock              # pinned dependencies snapshot
└── README.md                           # this file
```

The slider checkpoints produced here are saved under
[`sdxl/trained_sliders/sliders/`](../sliders/) (`.pt` / `.safetensors`).
The final sliders used by the paper runs are committed under `general/`
(un-anchored) and `anchored/` (subject-anchored) via a whitelist in
`.gitignore`; any other checkpoint produced by training is git-ignored
and consumed by the downstream tasks locally.

## How a slider is defined

Each slider is fully described by three files:

1. A **prompt YAML** under `prompts/new_prompt/<slider>.yaml`. Every YAML
   entry has four fields used by the Concept Sliders objective:

   | field | meaning |
   |---|---|
   | `target` | subject the LoRA is conditioned on; the LoRA is active on this prompt. |
   | `positive` | positive pole of the concept direction. |
   | `unconditional` | negative pole of the concept direction (not the empty CFG unconditional). |
   | `neutral` | base point used by the loss; conventionally equal to `target`. |
   | `action` | `enhance` for every entry used in the paper. |
   | `guidance_scale` | scale of the learned direction inside the loss (not CFG guidance). |
   | `resolution` / `batch_size` | training resolution / per-step batch size. |

   The semantic direction learned is `(positive - unconditional)`,
   applied as a shift from `neutral`, and active only when conditioned
   on `target`. With ``target = "woman"`` (subject-anchored) the slider
   is supervised to fire only on female subjects; with ``target = "person"``
   (un-anchored) it fires on any human. The paper §4.1 walks through the
   prompt patterns for both variants.

2. A **config YAML** under `configs/<slider>.yaml` that points to the
   prompt YAML and sets the model id, the LoRA hyperparameters (rank,
   alpha, training method) and the optimisation settings. The four
   subject-anchored sliders reported in the paper all use the same
   hyperparameters (rank 4, alpha 1.0, training method `noxattn`,
   AdamW with lr 2e-4 constant, 1000 iterations, bfloat16) so that the
   comparison against the un-anchored baseline isolates the prompt
   recipe.

3. A **SLURM template** under `jobs/new_slurm/train_<slider>.slurm` that
   activates the training environment and invokes
   `scripts/train.py --config_file <slider>.yaml`.

The prompt-vs-config split keeps the slider definition (which is what
§4.1 of the paper modifies) decoupled from the optimisation knobs (which
are held constant for the reported runs).

## Usage

From the repository root:

```bash
# Train one slider directly:
python sdxl/trained_sliders/training/scripts/train.py \
       --config_file sdxl/trained_sliders/training/configs/age_woman_sdxl_v1.yaml

# Or via the bundled SLURM template:
sbatch sdxl/trained_sliders/training/jobs/new_slurm/train_age_woman_sdxl_v1.slurm
```

The output is saved under
`outputs/<slider>_alpha<alpha>_rank<rank>_<method>/`
(local to the training folder), with intermediate checkpoints written
every `save.per_steps` steps. The trained slider then needs to be moved
to `sdxl/trained_sliders/sliders/` to be picked up by the evaluation
pipelines.

A typical training run on a single A100-class GPU takes ~45-75 min for
1000 iterations at 512x512.

## Adding a new slider

For a new slider called `<exp>`:

1. Create `prompts/new_prompt/<exp>.yaml` with the four-prompt
   definition. Several entries can be listed and the training loop
   samples one of them uniformly at random per step (this is how the
   subject-anchored variants of the paper combine "enhance on the
   target subject" with "preserve on the complementary subject" via
   degeneracy of the Concept Sliders objective).
2. Create `configs/<exp>.yaml` by copying an existing config and editing
   only `prompts_file`, `save.name` and `save.path`.
3. Create `jobs/new_slurm/train_<exp>.slurm` by copying an existing
   SLURM template and editing the `--job-name` and the `--config_file`.

No code changes are required for a new slider.

## Two preservation routes

The paper §4.1 describes a counter-anchor pattern with
``target == positive == unconditional``: when those fields are equal,
the Concept Sliders objective degenerates into
``MSE(LoRA_on, LoRA_off)``, i.e. pure preservation. The YAML-trick
variant exploits this by mixing enhance entries (on the target subject)
and counter-anchor entries (on the complementary subject) inside the
same prompt set. This is the route used for every subject-anchored
slider reported in the paper.

`scripts/train_with_preservation.py` is an alternative that decouples
the two terms: it loads two prompt sets (the man-anchored enhance
set and a woman-anchored preservation set) and combines them with a
configurable `--lambda_pres` coefficient. Both routes are kept in the
repository; only the YAML-trick was used for the paper because it
needs no `train.py` modification.

## Reproducibility

Each slider is fully defined by the triple
``prompts/new_prompt/<exp>.yaml`` + ``configs/<exp>.yaml`` +
``jobs/new_slurm/train_<exp>.slurm`` plus the repo commit. The
`scripts/` directory is a self-contained copy of the upstream
Concept Sliders training files, so future refactors of `sdxl/core/`
will not silently change the behaviour of past runs.
