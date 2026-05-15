#!/bin/bash
#SBATCH --job-name=sel_fur_p3
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=sdxl/tasks/selectivity/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/selectivity/logs/%x_%j.err

# =============================================================================
# eval_furlength_phase3.slurm  (selectivity task)
# =============================================================================
# Phase 3 — apply two furlength sliders GLOBALLY (no spatial mask):
#   - SPECIFICO: furlength_dog → deve modificare solo il cane
#   - GENERAL:  furlength_pet → baseline, modifies every subject
#
# Selectivity is computed afterwards by eval_selectivity.py using
# mask_target.png (cane) e mask_nontarget.png (gatto) create in fase 2.
#
# ⚠️  IMPOSTA I PATH DEI TUOI SLIDER QUI:
# =============================================================================

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_sdxl_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

MODEL_ID="$FERT_HF_CACHE/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
SLIDERS_ROOT="sdxl/trained_sliders/sliders"

# ⚠️  IMPOSTA IL PATH DEL TUO SLIDER SPECIFICO QUI (furlength_dog):
SLIDER_SPECIFIC="${SLIDERS_ROOT}/furlength_dog_sdxl_v1_alpha1.0_rank4_noxattn/furlength_dog_sdxl_v1_alpha1.0_rank4_noxattn_last.safetensors"

# General slider (furlength_pet — also used in masked_lora):
SLIDER_GENERAL="${SLIDERS_ROOT}/furlength_pet_sdxl_v1_alpha1.0_rank4_noxattn/furlength_pet_sdxl_v1_alpha1.0_rank4_noxattn_last.safetensors"

RUNS_ROOT="sdxl/tasks/selectivity/runs"

echo "================================================================"
echo "=== PHASE 3: selectivity/furlength  (20 runs x 2 sliders x 3 scales) ==="
echo "================================================================"
nvidia-smi
echo ""

for i in $(seq 1 20); do
    RUN_NUM=$(printf "%02d" "$i")
    RUN_ID="eval_furlength_${RUN_NUM}"
    RUN_DIR="${RUNS_ROOT}/${RUN_ID}"

    if [ ! -d "$RUN_DIR" ]; then
        echo "[SKIP] Run dir non trovata: ${RUN_DIR}"
        continue
    fi
    if [ ! -f "${RUN_DIR}/base.png" ]; then
        echo "[SKIP] base.png assente (fase 1 non eseguita): ${RUN_DIR}"
        continue
    fi

    echo "==== ${RUN_ID} ===="

    echo "  [specifico] furlength_dog"
    python sdxl/tasks/selectivity/scripts/03_apply_slider.py \
        --run_dir        "$RUN_DIR" \
        --slider_path    "$SLIDER_SPECIFIC" \
        --slider_scales  1.0 2.0 3.0 \
        --output_prefix  "edited_furlength_specific" \
        --rank           4 \
        --start_noise    750 \
        --model_id       "$MODEL_ID" \
        --device         cuda \
        --dtype          float16

    echo "  [generale] furlength_pet"
    python sdxl/tasks/selectivity/scripts/03_apply_slider.py \
        --run_dir        "$RUN_DIR" \
        --slider_path    "$SLIDER_GENERAL" \
        --slider_scales  1.0 2.0 3.0 \
        --output_prefix  "edited_furlength_general" \
        --rank           4 \
        --start_noise    750 \
        --model_id       "$MODEL_ID" \
        --device         cuda \
        --dtype          float16

    echo ""
done

echo "================================================================"
echo "=== PHASE 3 COMPLETED: selectivity/furlength                ==="
echo "================================================================"
echo "Per ogni run trovi:"
echo "  edited_furlength_specific_s1.0.png / s2.0.png / s3.0.png"
echo "  edited_furlength_general_s1.0.png  / s2.0.png / s3.0.png"
echo ""
echo "Then run (after phase 2 masks):"
echo "  python metrics/eval_selectivity.py --concept furlength --device cuda"
