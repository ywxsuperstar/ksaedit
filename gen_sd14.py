import os
from SDLens import HookedStableDiffusionPipeline
from training.k_sparse_autoencoder import SparseAutoencoder
from utils.hooks import add_feature_on_text_prompt, minus_feature_on_text_prompt
import torch
from tqdm.auto import tqdm
import argparse
import pandas as pd 
import time


def parse_args():
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/path/to/model/stable-diffusion-v1-4", 
    )
    parser.add_argument(
        "--start_iter",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end_iter",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=-1,
    )
    parser.add_argument(
        "--concept_steer",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="creat_data/eval.csv",
    )
    parser.add_argument(
        "--checkpoints_sae",
        type=str,
        default="",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    return parser.parse_args()
    
# CFG hook: +steering on cond, -steering on uncond
def modulate_hook_prompt(sae, steering_feature, block):  
    call_counter = {"count": 0}
    
    def hook_function(*args, **kwargs):
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            return add_feature_on_text_prompt(sae,steering_feature, *args, **kwargs)
        else:
            return minus_feature_on_text_prompt(sae,steering_feature,*args, **kwargs)

    return hook_function

# Manipulate text encoder feature to generate image
def activation_modulation_across_prompt(blocks_to_save, steer_prompt, strength, steps, guidance_scale, seed):
    output, cache = pipe.run_with_cache(  
        steer_prompt,
        positions_to_cache=blocks_to_save,
        save_input=True,
        save_output=True,
        num_inference_steps=1,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device="cpu").manual_seed(seed)
    )
    diff = cache['output'][blocks_to_save[0]][:,0,:]
    diff= diff.squeeze(0)

    with torch.no_grad():
        activated = sae.encode_without_topk(diff)
    mask = activated * (strength)

    to_add = mask @ sae.decoder.weight.T  # [77, 768]
    steering_feature = to_add  # [77, 768]
    
    if strength == 0:
        output = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance_scale,
                    generator=torch.Generator(device="cpu").manual_seed(seed))
    else:
        output = pipe.run_with_hooks(
            prompt,
            position_hook_dict = {
                block: modulate_hook_prompt(sae, steering_feature, block)
                for block in blocks_to_save
            },
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cpu").manual_seed(seed)
        )

    return output.images[0]
    
args = parse_args()
dtype = torch.float32
pipe = HookedStableDiffusionPipeline.from_pretrained(
    args.pretrained_model_name_or_path, torch_dtype=dtype)
pipe.set_progress_bar_config(disable=True)
pipe.to(args.device)

blocks_to_save = ['text_encoder.text_model.encoder.layers.10']   
sae = SparseAutoencoder.load_from_disk(os.path.join(args.checkpoints_sae, 'final')).to(args.device, dtype=dtype) 

num_inference_steps = 50  # denoising steps
guidance_scale = args.guidance_scale
batch_size = 1
outdir = args.outdir 

if not os.path.exists(outdir):
    os.makedirs(outdir)

data = pd.read_csv(args.prompt)
try: 
    prompts = data['prompt'].to_numpy()
except:
    prompts = data['adv_prompt'].to_numpy()

try:
    seeds = data['evaluation_seed'].to_numpy()
except:
    try:
        seeds = data['sd_seed'].to_numpy()
    except:
        seeds = [42 for i in range(len(prompts))]

try: 
    guidance_scales = data['evaluation_guidance'].to_numpy()
except:
    try:
        guidance_scales = data['sd_guidance_scale'].to_numpy()
    except:
        guidance_scales = [7.5 for i in range(len(prompts))]

i = args.start_iter
n_samples = len(data)

avg_time = 0.0
progress_bar = tqdm(total=min(n_samples, args.end_iter) - i, desc="Processing Samples")

while i < n_samples and i< args.end_iter:   
    
    torch.cuda.empty_cache()
    try:
        seed = int(seeds[i])
    except:
        seed = int(seeds[i][0])
    prompt = [prompts[i]] 
    guidance_scale = float(guidance_scales[i])
    print(prompt, seed, guidance_scale)

    if i+ batch_size > n_samples:
        batch_size = n_samples - i
    start_time = time.time()
    
    with torch.no_grad():
        image = activation_modulation_across_prompt(blocks_to_save, args.concept_steer, args.strength, num_inference_steps, guidance_scale, seed )
        for j in range(batch_size):
            end_time = time.time()
            avg_time += end_time - start_time
            image.save(f"{outdir}/{i+j}.png")            
    i += batch_size 
    progress_bar.update(batch_size)  

progress_bar.close()  
avg_time = avg_time/float(i)
print(f'avg_time: {avg_time}')