import os
import time
import argparse

import pandas as pd
import torch
from tqdm.auto import tqdm

from SDLens import HookedStableDiffusionXLPipeline
from training.k_sparse_autoencoder import SparseAutoencoder
from utils.hooks import add_feature_on_text_prompt, minus_feature_on_text_prompt

def parse_args():
    parser = argparse.ArgumentParser(description="Compositional attribute generation via summed SAE steering vectors.")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="/path/to/model/stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--end_iter", type=int, default=10000)
    parser.add_argument("--outdir", type=str, default="")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--strength", type=str, default="0.15,0.15",
                        help="Comma-separated strengths, one per concept (e.g. '0.1,0.2').")
    parser.add_argument("--concept_steer", type=str, default=None,
                        help="Semicolon-separated concept prompts (e.g. 'happy, big smile; very old, wrinkly').")
    parser.add_argument("--prompt", type=str, default="creat_data/eval-sdxl.csv")
    parser.add_argument("--checkpoints_sae", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def modulate_hook_prompt(sae, steering_feature, block):
    call_counter = {"count": 0}

    def hook_function(*args, **kwargs):
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            return add_feature_on_text_prompt(sae, steering_feature, *args, **kwargs)
        else:
            return minus_feature_on_text_prompt(sae, steering_feature, *args, **kwargs)

    return hook_function


def activation_modulation_across_multiple_prompts(
    pipe, sae, image_prompt, blocks_to_save, steer_prompts, strengths, steps, guidance_scale, seed
):
    """Sum SAE steering vectors from each concept prompt, then one hooked generation on image_prompt."""
    if not steer_prompts or all(s == 0 for s in strengths):
        output = pipe(
            image_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )
        return output.images[0]

    if len(steer_prompts) != len(strengths):
        raise ValueError(
            f"concept_steer has {len(steer_prompts)} concepts (split by ';') "
            f"but strength has {len(strengths)} values (split by ','). Counts must match."
        )

    combined_feature = None
    for steer_prompt, strength in zip(steer_prompts, strengths):
        _, cache = pipe.run_with_cache(
            steer_prompt,
            positions_to_cache=blocks_to_save,
            save_input=True,
            save_output=True,
            num_inference_steps=1,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )

        diff1 = cache["output"][blocks_to_save[0]][:, 0, :].squeeze(0)  # [77, 768]
        diff2 = cache["output"][blocks_to_save[1]][:, 0, :].squeeze(0)  # [77, 1280]
        diff = torch.cat([diff1, diff2], dim=-1)  # [77, 2048]

        with torch.no_grad():
            activated = sae.encode_without_topk(diff)
        to_add = (activated * strength) @ sae.decoder.weight.T

        if combined_feature is None:
            combined_feature = to_add
        else:
            combined_feature = combined_feature + to_add

    output = pipe.run_with_hooks(
        image_prompt,
        position_hook_dict={
            block: modulate_hook_prompt(sae, combined_feature, block)
            for block in blocks_to_save
        },
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    return output.images[0]


# ---------------------------------------------------------------------------
args = parse_args()

dtype = torch.float32
pipe = HookedStableDiffusionXLPipeline.from_pretrained(
    args.pretrained_model_name_or_path, torch_dtype=dtype)
pipe.set_progress_bar_config(disable=True)
pipe.to(args.device)

blocks_to_save = [
    "text_encoder.text_model.encoder.layers.10",
    "text_encoder_2.text_model.encoder.layers.28",
]
sae = SparseAutoencoder.load_from_disk(
    os.path.join(args.checkpoints_sae, "final")
).to(args.device, dtype=dtype)

num_inference_steps = 50
batch_size = 1
outdir = args.outdir
os.makedirs(outdir, exist_ok=True)

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

strengths = [float(s.strip()) for s in args.strength.split(",") if s.strip()] if args.strength else [0.0]
steer_prompts = (
    [s.strip() for s in args.concept_steer.split(";") if s.strip()]
    if args.concept_steer else []
)

i = args.start_iter
n_samples = len(data)
avg_time = 0
progress_bar = tqdm(total=min(n_samples, args.end_iter) - i, desc="Processing Samples")

while i < n_samples and i < args.end_iter:
    torch.cuda.empty_cache()
    try:
        seed = int(seeds[i])
    except:
        seed = int(seeds[i][0])
    prompt = [prompts[i]]
    guidance_scale = float(guidance_scales[i])

    if i + batch_size > n_samples:
        batch_size = n_samples - i

    start_time = time.time()
    with torch.no_grad():
        image = activation_modulation_across_multiple_prompts(
            pipe, sae, prompt, blocks_to_save, steer_prompts, strengths,
            num_inference_steps, guidance_scale, seed,
        )
        for j in range(batch_size):
            image.save(f"{outdir}/{i+j}.png")
            print(f"Saved image: {outdir}/{i+j}.png")
    end_time = time.time()
    avg_time += end_time - start_time
    i += batch_size
    progress_bar.update(batch_size)

progress_bar.close()
avg_time = avg_time / float(i)
print(f'avg_time: {avg_time}')
