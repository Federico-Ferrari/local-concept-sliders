#!/bin/bash
#SBATCH --job-name=masked_painterly_p3
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=sdxl/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/masked_lora/logs/%x_%j.err

# Prerequisiti: base.png + init_latents.pt + mask.png in runs/masked_painterly_01/
# mask.png = maschera della regione da rendere pittorica (es: cielo o edificio)

set -euo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_sdxl_env

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub

RUN_ID="masked_painterly_01"
RUN_DIR="sdxl/tasks/masked_lora/runs/${RUN_ID}"
SLIDER="sdxl/trained_sliders/sliders/painterly_sdxl_v1_alpha1.0_rank4_noxattn/painterly_sdxl_v1_alpha1.0_rank4_noxattn_last.safetensors"
MODEL_ID="$FERT_HF_CACHE/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"

echo "=== FASE 3: MASKED EDIT — painterly — ${RUN_ID} ==="
nvidia-smi
echo ""
ls -lh "$RUN_DIR/"

for SCALE in 1.5 2.0 3.0; do
    echo "--- Scale ${SCALE} ---"
    python sdxl/tasks/masked_lora/scripts/03_masked_edit.py \
        --run_dir "$RUN_DIR" \
        --slider_path "$SLIDER" \
        --slider_scale "$SCALE" \
        --rank 4 \
        --start_noise 750 \
        --model_id "$MODEL_ID" \
        --device cuda \
        --dtype float16 \
        --mask_name mask.png \
        --output_name "edited_painterly_s${SCALE}.png"
done

echo "=== COMPLETATO ===" && ls -lh "$RUN_DIR/"
