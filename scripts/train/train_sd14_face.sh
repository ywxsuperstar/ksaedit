PATHS_TO_LATENTS="feature/face_attr_sd14"

BLOCK_NAME="text_encoder.text_model.encoder.layers.10"
checkpoint_path=Checkpoints_oneattr_d64k64
D_MODEL="768"

python train_ksae.py \
    --paths_to_latents "$PATHS_TO_LATENTS" \
    --block_name "$BLOCK_NAME" \
    --checkpoint_path "$checkpoint_path" \
    --d_in "$D_MODEL" \
    --total_training_tokens "400_000_000" \
    --expansion_factor 64 \
    --k 64 
