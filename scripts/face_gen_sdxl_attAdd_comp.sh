#!/bin/bash
# "smile_strength,old_strength"
declare -a STRENGTH_PAIRS=(
    "0.20,0.25"
    "0.25,0.20"
    "0.25,0.25"
)

CONCEPT_STEER="happy, big smile; older, old, very old, very age, white hair"
PROMPT="creat_data/eval-sdxl.csv"
CheckpointsSAE="/path/to/checkpoints"

for strength in "${STRENGTH_PAIRS[@]}"; do
    IFS=',' read -r a b <<< "$strength"
    OUTDIR="outImg/sdxl_intact_oneAtt52_block10and28/attAdd/smile${a}_old${b}/"

    echo "Running with smile=$a, old=$b"
    python comp_gen_sdxl_attAdd_args.py \
        --concept_steer "$CONCEPT_STEER" \
        --start_iter 0 \
        --prompt "$PROMPT" \
        --strength "$strength" \
        --checkpoints_sae "$CheckpointsSAE" \
        --outdir "$OUTDIR"
done
