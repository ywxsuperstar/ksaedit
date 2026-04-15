import os
from SDLens import HookedStableDiffusionPipeline,HookedStableDiffusionXLPipeline,HookedAutoPipelineForImage2Image
from training.k_sparse_autoencoder import SparseAutoencoder
from utils.hooks import add_feature_on_text_prompt, do_nothing, minus_feature_on_text_prompt
# from utils import add_feature_on_text_prompt, do_nothing, minus_feature_on_text_prompt
import torch
from tqdm.auto import tqdm
import argparse
import pandas as pd 
import time
from PIL import Image
import numpy as np

import sys
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DDIMScheduler,
)

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output
from diffusers.utils.torch_utils import randn_tensor
sys.path.append('ksaedit/ReNoiseInversion')
from ReNoiseInversion.src.config import RunConfig, Scheduler_Type
from ReNoiseInversion.src.eunms import Model_Type, Scheduler_Type, Gradient_Averaging_Type, Epsilon_Update_Type
from ReNoiseInversion.src.enums_utils import model_type_to_size, get_pipes,scheduler_type_to_class
from ReNoiseInversion.main import inversion_callback, inference_callback

torch.set_float32_matmul_precision('high')
DEVICE = 'cuda:4'
dtype = torch.float16
device = 'cuda:4' if torch.cuda.is_available() else 'cpu'


def activation_modulation_across_prompt(config,sae,pipe,image_latents, blocks_to_save, steer_prompt, prompt, strength, steps, guidance_scale, seed):
    
    output, cache = pipe.run_with_cache(
        steer_prompt,
        positions_to_cache=blocks_to_save,
        save_input=True,
        save_output=True,
        num_inference_steps=1,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        image=image_latents
    )

    diff1 = cache['output'][blocks_to_save[0]][:, 0, :]  # 获取第一块特征
    diff2 = cache['output'][blocks_to_save[1]][:, 0, :]  # 获取第二块特征
    diff1 = diff1.squeeze(0)
    diff2 = diff2.squeeze(0)
    diff = torch.cat([diff1, diff2], dim=-1)
    diff = diff.to(torch.float16)

    with torch.no_grad():
        activated = sae.encode_without_topk(diff)
    mask = activated * strength
    to_add = mask @ sae.decoder.weight.T
    steering_feature = to_add  # [77, 2048]

    output = pipe.run_with_hooks(
        prompt,
        position_hook_dict={
            block: modulate_hook_prompt(sae, steering_feature, block)
            for block in blocks_to_save
        },
        num_inference_steps=config.num_inference_steps,
        callback_on_step_end=inference_callback,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        image=image_latents,
        strength=config.inversion_max_step,
        denoising_start=1.0 - config.inversion_max_step,
        guidance_scale=1.0,
    )
    return output.images[0]



def parse_args():
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/path/to/model/sdxl-turbo", 
    )
    parser.add_argument(
        "--guidance",
        type=str,
        default=None,
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
        default="",
    )
    parser.add_argument(
        "--checkpoints_sae",
        type=str,
        default="",
    )
    parser.add_argument(
        "--reference_image",
        type=str,
        default="",
        help="Path to reference image for image-conditioned generation"
    )
    return parser.parse_args()

def modulate_hook_prompt(sae, steering_feature, block):  # 在第一次调用时添加特征，后续调用时减去特征
    call_counter = {"count": 0}
    
    def hook_function(*args, **kwargs):
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            return add_feature_on_text_prompt(sae,steering_feature, *args, **kwargs)
        else:
            return minus_feature_on_text_prompt(sae,steering_feature,*args, **kwargs)

    return hook_function


def main():

    # Default ReNoise settings, only inversion strength decreased slightly to improve consistency (similar to delayed application in our paper)
    model_type = Model_Type.SDXL_Turbo
    scheduler_type = Scheduler_Type.EULER
    first_step_range_end = 5
    number_of_renoising_iterations = 9
    inersion_strength = 0.3 # 1.0  #Attcontrol中设0.8, 越大表示保留原图的信息越多，编辑越难
    avg_gradients_type = Gradient_Averaging_Type.ON_END
    first_step_range = (0, 5)
    rest_step_range = (8, 10)
    # lambda_ac = 20.0
    # lambda_kl = 0.055
    lambda_ac = 0.0
    lambda_kl = 0.0
    update_epsilon_type = Epsilon_Update_Type.OPTIMIZE
    config = RunConfig(model_type = model_type,
        num_inference_steps = 4,
        num_inversion_steps = 4, 
        guidance_scale = 0.0,
        max_num_aprox_steps_first_step = first_step_range_end+1,
        num_aprox_steps = number_of_renoising_iterations,
        inversion_max_step = inersion_strength,
        gradient_averaging_type = avg_gradients_type,
        gradient_averaging_first_step_range = first_step_range,
        gradient_averaging_step_range = rest_step_range,
        scheduler_type = scheduler_type,
        num_reg_steps = 4,
        num_ac_rolls = 5,
        lambda_ac = lambda_ac,
        lambda_kl = lambda_kl,
        update_epsilon_type = update_epsilon_type,
        do_reconstruction = True
        # prompt = prompt
    )
    image_size = model_type_to_size(Model_Type.SDXL_Turbo)
    def get_pipes_my():
        pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=DEVICE) 
        pipe_inversion.safety_checker = None
        pipe_inference.safety_checker = None
        # Inversion code, adapted from https://huggingface.co/spaces/garibida/ReNoise-Inversion/blob/main/main.py

        if config.scheduler_type == Scheduler_Type.EULER or config.scheduler_type == Scheduler_Type.LCM or config.scheduler_type == Scheduler_Type.DDPM:
            g_cpu = torch.Generator().manual_seed(42) #7865
            img_size = model_type_to_size(config.model_type)
            VQAE_SCALE = 8
            latents_size = (1, 4, img_size[0] // VQAE_SCALE, img_size[1] // VQAE_SCALE)
            # num_inv_steps = len(pipe_inversion.scheduler.sigmas)
            noise = [randn_tensor(latents_size, dtype=torch.float16, device=torch.device(DEVICE), generator=g_cpu) for i in range(config.num_inversion_steps)]
            pipe_inversion.scheduler.set_noise_list(noise)
            pipe_inference.scheduler.set_noise_list(noise)
            pipe_inversion.scheduler_inference.set_noise_list(noise)
        else:
            raise NotImplementedError() 

        if config.save_gpu_mem:
            pipe_inference.to("cpu")
            pipe_inversion.to(DEVICE)
        pipe_inversion.cfg = config
        pipe_inference.cfg = config

        return pipe_inversion, pipe_inference, noise

    args = parse_args()
    guidance = args.guidance
    
    blocks_to_save = ['text_encoder.text_model.encoder.layers.10', 'text_encoder_2.text_model.encoder.layers.28']     
    sae = SparseAutoencoder.load_from_disk(os.path.join(args.checkpoints_sae, 'final')).to(device, dtype=torch.float16) 
    num_inference_steps = 10  # denoising steps  50
    guidance_scale = args.guidance_scale  
    torch.cuda.manual_seed_all(42)
    batch_size = 1
    outdir = args.outdir 

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    n_samples = args.end_iter
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

    try:
        case_number = data['case_number'].to_numpy()
    except:
        case_number = [0 for i in range(len(prompts))]


    i = args.start_iter
    n_samples = len(data)

    avg_time = 0.0
    progress_bar = tqdm(total=min(n_samples, args.end_iter) - i, desc="Processing Samples")

    parent_dir = os.path.dirname(outdir)
    dir_path = os.path.join(parent_dir, "inversion")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    
    while i < n_samples and i< args.end_iter:   
        
        torch.cuda.empty_cache()
        pipe_inversion, pipe_inference,noise = get_pipes_my()
        try:
            seed = int(seeds[i])
        except:
            seed = int(seeds[i][0])
        prompt = [prompts[i]]
        
        def simple_ddim_inversion(ref_image, prompt):
            res = pipe_inversion(
                prompt = prompt,
                num_inversion_steps = config.num_inversion_steps,
                num_inference_steps = config.num_inference_steps,
                image = ref_image.convert('RGB').resize(image_size),
                guidance_scale = config.guidance_scale,
                opt_iters = config.opt_iters,
                opt_lr = config.opt_lr,
                callback_on_step_end = inversion_callback,
                strength = config.inversion_max_step,
                denoising_start = 1.0-config.inversion_max_step,
                opt_loss_kl_lambda = config.loss_kl_lambda,
                num_aprox_steps = config.num_aprox_steps
            )
            latents = res[0][0]
            # all_latents = res[1]

            print('Inverted Image')
            if config.save_gpu_mem:
                pipe_inference.to(DEVICE)
                pipe_inversion.to("cpu")
            img_inversion = pipe_inference(
                prompt = prompt,
                num_inference_steps=config.num_inference_steps,
                negative_prompt=config.prompt,
                callback_on_step_end=inference_callback,
                image=latents,
                strength=config.inversion_max_step,
                denoising_start=1.0 - config.inversion_max_step,
                guidance_scale=1.0
            ).images[0]

            img_inversion.save(f'{dir_path}/{i}.png')
            img_inversion = None
            return latents, img_inversion  # return final x_t

        torch.cuda.empty_cache()
        guidance_scale = float(guidance_scales[i])
        print(prompt, seed, guidance_scale)
        torch.cuda.manual_seed_all(seed)

        if i+ batch_size > n_samples:
            batch_size = n_samples - i
        start_time = time.time()
        
        # with torch.no_grad():
        # ref_image_path = os.path.join(args.reference_image,f"{int(case_number[i]):012}.jpg")  # COCO val
        ref_image_path = os.path.join(args.reference_image,f"{int(case_number[i])}.png")  # COCO val
        if not os.path.exists(ref_image_path):
            raise FileNotFoundError(f"ref img not exit:{ref_image_path}")
        
        config.prompt = prompt[0]
        ref_image = Image.open(ref_image_path)
        image_latents,image_rec = simple_ddim_inversion(ref_image, prompt[0])
        # del pipe_inversion.unet, pipe_inversion.vae
        # del pipe_inference.unet, pipe_inference.vae
        scheduler_pipe = pipe_inversion.scheduler
        del pipe_inversion, pipe_inference
        torch.cuda.empty_cache()
        # pipe = HookedAutoPipelineForImage2Image.from_pretrained(
        #     "/path/to/model/sdxl-turbo", safety_checker = None,
        #     torch_dtype=torch.float32,scheduler=pipe_inversion.scheduler) # scheduler=scheduler
        pipe = HookedAutoPipelineForImage2Image.from_pretrained(
            "/path/to/model/sdxl-turbo", safety_checker = None,
            torch_dtype=torch.float32,scheduler=scheduler_pipe) # scheduler=scheduler
        scheduler_class = scheduler_type_to_class(scheduler_type)
        pipe.scheduler            = scheduler_class.from_config(pipe.scheduler.config)
        pipe.scheduler_inference  = scheduler_class.from_config(pipe.scheduler.config)

        pipe.scheduler.add_noise = lambda init_latents, noise, timestep: init_latents
        pipe.scheduler.add_noise = lambda init_latents, noise, timestep: init_latents
        pipe.scheduler_inference.add_noise = lambda init_latents, noise, timestep: init_latents
        pipe.scheduler.set_noise_list(noise)
        pipe.scheduler_inference.set_noise_list(noise)
        pipe.set_progress_bar_config(disable=True)
        pipe.to(device)
        image = activation_modulation_across_prompt(config,sae,pipe, image_latents,blocks_to_save, args.concept_steer, prompt[0], args.strength, num_inference_steps, guidance_scale, seed )
        pipe.to("cpu")
        torch.cuda.empty_cache()
        for j in range(batch_size):
            end_time = time.time()
            avg_time += end_time - start_time
            image.save(f"{outdir}/{i+j}.png")            
        i += batch_size 
        progress_bar.update(batch_size)  
        torch.cuda.empty_cache()

    progress_bar.close()  
    avg_time = avg_time/float(i)
    print(f'avg_time: {avg_time}')


if __name__ == "__main__":
    main()