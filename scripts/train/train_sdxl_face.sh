D_MODEL="2048"
block1=10  
block2=28  
EXPANSION_FACTOR=16  
K=128
AUXK=256
auxk_coef='0.1'
PATHS_TO_LATENTS="feature/face_attr_sdxl"
BLOCK_NAME="text_encoder.text_model.encoder.layers.${block1}.${block2}"
checkpoint_path="Checkpoints/sdxl_oneAtt52/ckpt_block${block1}and${block2}_e${EXPANSION_FACTOR}k${K}kaux${AUXK}_Laux${auxk_coef}_lr4e-4"

python train_ksae.py \
    --paths_to_latents "$PATHS_TO_LATENTS" \
    --block_name "$BLOCK_NAME" \
    --checkpoint_path "$checkpoint_path" \
    --d_in "$D_MODEL" \
    --total_training_tokens "400_000_000" \
    --expansion_factor "$EXPANSION_FACTOR" \
    --k "$K" \
    --auxk "$AUXK" \
    --auxk_coef "$auxk_coef" \
    --lr 0.0004 