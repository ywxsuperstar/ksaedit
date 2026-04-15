
import os
import pandas as pd
import sys
import datetime
import json
import torch
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from SDLens.hooked_sd_pipeline import HookedStableDiffusionPipeline
import fire
import numpy as np

def main(save_path='feature/face_attr_sd14', start_at=0, finish_at=90000, chunk_size=1000):
    blocks_to_save = ['text_encoder.text_model.encoder.layers.10' ]      
    block = 'text_encoder.text_model.encoder.layers.10'

    csv_filepaths = ["creat_data/prompt_attr_52.csv"]

    data_frames = [pd.read_csv(filepath) for filepath in csv_filepaths]
    data = pd.concat(data_frames, ignore_index=True)
    prompts = data['prompt'].to_numpy()
    
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
            guidance_scales =data['sd_guidance_scale'].to_numpy()
        except:
            guidance_scales = [7.5 for i in range(len(prompts))]

    # Initialize pipeline
    dtype = torch.float32
    pipe = HookedStableDiffusionPipeline.from_pretrained("ksaedit/stable-diffusion-v1-4", torch_dtype=dtype)                                                       
    pipe.to('cuda:0')
    pipe.set_progress_bar_config(disable=True)
    
    # Create save path and metadata
    ct = datetime.datetime.now()
    save_path = os.path.join(save_path, str(ct))
    os.makedirs(save_path, exist_ok=True)

    data_tensors = []
    metadata = []
    chunk_idx = 0
    chunk_start_idx = start_at

    # Processing prompts
    for num_document in tqdm(range(len(prompts)), desc="Processing Prompts", unit="prompt"):
        if num_document < start_at:
            continue
        if num_document >= finish_at:
            break

        kwargs_to_save = {
            'prompt': prompts[num_document],
            'positions_to_cache': blocks_to_save,
            'save_input': True,
            'save_output': True,
            'num_inference_steps': 1,
            'guidance_scale': guidance_scales[num_document],
            'seed': int(seeds[num_document]),
            'output_type': 'pil',
        }
        _, cache = pipe.run_with_cache(**kwargs_to_save)
        
        sample_output = cache['output'][blocks_to_save[0]][:,0].cpu()
        data_tensors.append(sample_output)

        # Store metadata
        metadata.append({
            "sample_id": num_document,
            "gen_args": kwargs_to_save
        })

        # Save chunk if it reaches the specified size
        if len(data_tensors) >= chunk_size:
            chunk_end_idx = chunk_start_idx + len(data_tensors) - 1
            save_chunk(data_tensors, metadata, save_path, chunk_start_idx, chunk_end_idx, chunk_idx, block)
            chunk_start_idx += len(data_tensors)
            data_tensors = []
            metadata = []
            chunk_idx += 1

    if data_tensors:
        chunk_end_idx = num_document
        save_chunk(data_tensors, metadata, save_path, chunk_start_idx, chunk_end_idx, chunk_idx, block)

    print(f"Data saved in chunks to {save_path}")


def save_chunk(data_tensors, metadata, save_path, start_idx, end_idx, chunk_idx, block):
    """Save a chunk of tensors and metadata with index tracking."""
    chunk_path = os.path.join(save_path, f'{block}_{start_idx:06d}_{end_idx:06d}.pt')
    metadata_path = os.path.join(save_path, f'metadata_{start_idx:06d}_{end_idx:06d}.json')

    # Stack tensors and save
    torch.save(torch.cat(data_tensors), chunk_path)

    # Save metadata as JSON
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4, default=lambda o: int(o) if isinstance(o, (np.integer, torch.Tensor)) else o)

    print(f"Saved chunk {chunk_idx}: {chunk_path}")

if __name__ == '__main__':
    fire.Fire(main)
