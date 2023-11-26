import argparse
import os
import random

import numpy as np
import pandas as pd

import torch

from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
import warnings
warnings.filterwarnings("ignore")

from src.experiments.py.demo import demo_model_editing
from src.bcf import BCFHyperParams
from src.util.globals import *
from src.bcf.bcf_main import get_context_templates

def parse_args():
    parser = argparse.ArgumentParser(
        prog='Clip Editor',
        description='A script for running and editing method'
                    'on CLIP and running it with bulk of concepts')
    parser.add_argument('--file', required=False, help='A file for running and testing the editing from', default='data/art_forget/artist_list_forget_10.csv')
    parser.add_argument('--algorithm', default='bcf', choices=['bcf','baseline'])
    parser.add_argument('--concept_type', default='artist_forget', choices=['object_forget', 'artist_forget', 'NSFW'])  
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--clip_model', default='openai/clip-vit-large-patch14')
    parser.add_argument('--v_similarity_metric', default='l2', choices=["l2", "cosine"])
    

    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def generate_with_seed(sd_pipeline, prompts, seed, output_path="./", image_params="", save_image=True):
    '''
    generate an image through diffusers 
    '''
    outputs = []
    generator = torch.Generator("cuda").manual_seed(seed)
    for prompt in prompts:
        print(prompt)
        image = sd_pipeline(prompt=prompt,generator=generator)['images'][0]
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except FileExistsError:
                pass
        if image_params != "":
            image_params = "_" + image_params

        image_name = f"{output_path}/{prompt}_seed_{seed}{image_params}.png"
        if save_image:
            image.save(image_name)
        print("Saved to: ", image_name)
        outputs.append((image, image_name))

    if len(outputs) == 1:
        return outputs[0]
    return outputs


def main():
    args = parse_args()
    print(args)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    seed = args.seed
    set_seed(seed)

    clip_model_name = args.clip_model
    
    sd_model_name = args.model
    sd_pipeline = StableDiffusionPipeline.from_pretrained(sd_model_name,safety_checker = None)
    # we disable the safety checker for test
    sd_pipeline = sd_pipeline.to(device)

    valid_set = pd.read_csv(args.file)

    path = "prompt"

    if args.algorithm == 'baseline':
        # baseline model is an unedited model, we directly use the prompt to generate the images
        for idx, raw_row in valid_set.iterrows():
            row = dict()
            for k,v in raw_row.items():
                row[k.lower()] = v.lower()

            templates =  get_context_templates(type = args.concept_type)
            prompts = [templates[i].format(row[path]) for i in range(3)]
            for prompt in prompts:
                generate_with_seed(sd_pipeline, [prompt], seed,
                                output_path=f"./images/results/{args.algorithm}/{args.concept_type}/{row[path]}")

    elif args.algorithm == "bcf":
        requests = []
        model = CLIPModel.from_pretrained(clip_model_name).to(device)
        processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        for idx, raw_row in valid_set.iterrows():
            row = dict()
            for k,v in raw_row.items():
                row[k.lower()] = v.lower()

            torch.cuda.empty_cache() 
            
            print(f"Editing: {row[path]}")
            
            requests.append(
                {
                    "prompt": row[path],
                    "subject": row[path],
                    "new_text": row['new'],
                    "true_text": row['old'],
                    "algorithm": args.algorithm
                }
            )
            # l2
            if args.v_similarity_metric:
                requests[idx]["similarity_metric"] = args.v_similarity_metric

        hparams_prefix,= "bcf"
        params_name = (
                HPARAMS_DIR
                / hparams_prefix
                / "openai_clip-vit-large-patch14-336.json"
        )
        hparams = BCFHyperParams.from_json(params_name)
        print("Loading from", params_name)
        print(hparams)

        model_new, _ = demo_model_editing(model, processor, requests,concept_type=args.concept_type, alg_name="bcf",
                                                        hparams=hparams)

        model.to('cpu')
        del model
        del processor
        model_new.text_model.dtype = torch.float32
        
        # Save edited model
        if len(requests) == 1 or (requests[0]["prompt"] == requests[1]["prompt"]):
            save_name = requests[0]["prompt"]
        else:
            save_name = len(requests)
        output_path = './model/{}/'.format(args.concept_type)
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except FileExistsError:
                pass
        output_path = os.path.join(output_path,"edited_text_encoder_{}_{}.pt".format(args.concept_type,save_name))
        torch.save(model_new.text_model.state_dict(),output_path)
        
        print("===============================================")
        print("edited model is successfully saved at {}".format(output_path))
        print("===============================================")
        
        sd_pipeline.text_encoder.text_model = model_new.text_model
        sd_pipeline = sd_pipeline.to(device)

        # post edited stable diffusion generations
        templates =  get_context_templates(type = args.concept_type)
        for i in range(len(requests[:10])):
            current_prompt = requests[i]["prompt"]
            prompts = [templates[j].format(current_prompt) for j in range(3)]
            for prompt in prompts:
                generate_with_seed(sd_pipeline, [prompt], seed,
                                output_path=f"./images/results/{args.algorithm}/{args.concept_type}/{current_prompt}")
    else:
        raise Exception(f"Unknown Algorithm: {args.algorithm}")
    
    # test the unedited concept:
    if args.concept_type == 'artist_forget':
        current_prompt = "Claude Monet"
        test_prompts = ['a river bank drawn by {} ',
                        'ships around the river bank in the style of {}',
                        'A famous artwork by {}'
                        ]
        prompts = [ test_prompts[j].format(current_prompt) for j in range(len(test_prompts))]
        for prompt in prompts:
            generate_with_seed(sd_pipeline, [prompt], seed,
                            output_path=f"./images/results/{args.algorithm}/{args.concept_type}/{current_prompt}")


if __name__ == "__main__":
    main()
