declare -a ATTRIBUTES=( 
    "smiling:happy face, grinning, big smile" # beaming
    "old:older, very old, very age, olderly, wrinkly, gray hair, aged skin"
    "blond-hair:blond-hair, blonde hair, golden hair"
    "Afro:afro, curly afro" 
    "wearing-lipstick:wearing-lipstick, wearing lipstick" 
    "heavy-makeup:heavy-makeup, bold makeup, striking makeup"  # Full Glam
    "wavy-hair:wavy-hair,curly hair, wavy, curly, curly hair, wavy hair"
    "eyeglasses:eyeglasses, wearing glasses, glasses"
    "wearing-necklace:wearing-necklace, wearing a necklace, necklace"
    "black-hair:black-hair, dark hair, deep black hair"      
    "bald:bald, no hair"
)


STRENGTHS=(0.25)
CheckpointsSAE="/path/to/checkpoints"
PROMPT="creat_data/eval-sdxl.csv"  

for attr_prompt in "${ATTRIBUTES[@]}"; do
    IFS=':' read -r attr prompt_extension <<< "$attr_prompt"  
    for strength in "${STRENGTHS[@]}"; do
        OUTDIR="outImg/sdxl_intact_oneAtt52_block10and28/e16k128_coef1div10/${attr}_${strength}"
        python gen_sdxl_oneAtt52_10and28_att2sae.py \
            --concept_steer "$prompt_extension" \
            --start_iter 0 \
            --prompt "$PROMPT" \
            --strength "$strength" \
            --checkpoints_sae "$CheckpointsSAE" \
            --outdir "$OUTDIR"
    done
done
