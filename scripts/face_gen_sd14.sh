CONCEPT_STEER="older, old, very old, very age"
PROMPT="creat_data/eval.csv"  # prompt csv file
OUTDIR="outImg/sd14/"
CheckpointsSAE="/path/to/checkpoints"   

python gen_sd14.py \
    --concept_steer "$CONCEPT_STEER" \
    --start_iter 0 \
    --prompt "$PROMPT" \
    --strength 0.25 \
    --checkpoints_sae "$CheckpointsSAE" \
    --outdir "$OUTDIR"  
 