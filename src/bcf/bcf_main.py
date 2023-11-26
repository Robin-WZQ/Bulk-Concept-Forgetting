from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPTokenizer

from src.util import nethook
from src.util.globals import *

from .compute_z import compute_z,get_module_input_output_at_words
from .compute_ks import compute_ks
from .bcf_hparams import BCFHyperParams
from .layer_stats import layer_stats

CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

def apply_bcf_to_model(
    model: CLIPModel,
    processor: CLIPProcessor,
    requests: List[Dict],
    hparams: BCFHyperParams,
    concept_type:str,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
) -> Tuple[CLIPModel, List[str]]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    if copy:
        model = deepcopy(model)

    weights_copy = {}

    # bcf main function
    deltas = execute_bcf(
        model, processor, requests, hparams, concept_type
    )

    # update the model weights
    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to("cuda"), val_mat.to("cuda")
            upd_matrix = val_mat @ key_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_bcf(
    model: CLIPModel,
    processor: CLIPProcessor,
    requests: Dict,
    hparams: BCFHyperParams,
    type:str,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the BCF update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    deltas = {}
    # Update target and print info
    requests = deepcopy(requests)
    print(f"Executing BCF algorithm for the update: {requests[0]['prompt']}, etc.")

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    print("weights.keys(): ", list(weights.keys()))

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    
    # Compute z for final layer
    context_templates = get_context_templates(type=type)
    z_layer = hparams.layers[-1]
    z_list = []
    
    clean_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')

    for request in tqdm(requests):
        cur_z = compute_z(
                        model,
                        processor,
                        request,
                        hparams,
                        z_layer,
                        context_templates,
                        clean_model,
                    )

        z_list.append(cur_z)

    zs = torch.stack(z_list, dim=1) #[768,NUM_requests]
    
    # Insert
    tok = processor.tokenizer
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(
                            model,
                            tok,
                            requests,
                            hparams, 
                            layer, 
                            context_templates
                        ).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"].replace(request["subject"], "{}") for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T # [768,NUM_requests]
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        # Load covariance matrix
        force_recompute = False
        # force_recompute = layer != hparams.layers[0]
        cov = get_cov(
            model.text_model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
        )

        # Compute update in double precision
        layer_ks, targets = (
            layer_ks.double(),
            targets.double(),
        )

        adj_k = torch.linalg.solve(
            (hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T).T,
            layer_ks,
        )
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = resid @ adj_k.T

        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
            )
            
        # Clear GPU memory
        cov.cpu()
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by BCF does not match original weight shape. "
            "Check for bugs in the code?"
        )

def get_cov(
    model: CLIPTextModel,
    tok: CLIPTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    # key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    stat = layer_stats(
        model,
        tok,
        layer_name,
        STATS_DIR,
        mom2_dataset,
        to_collect=["mom2"],
        sample_size=mom2_n_samples,
        precision=mom2_dtype,
        force_recompute=force_recompute,
    )
    key = stat.mom2.moment().float().to("cuda")

    return torch.inverse(key) if inv else key

def get_context_templates(type=str):
    if type == 'object_forget':
        templates = ["{} image",
                     "an Image of {}", 
                     "a painting of {}", 
                     "a strikingly realistic photo of {}", 
                     "a drawing of {}", 
                     "a beautiful photograph of {}", 
                     "{} shown in the room",
                     "realistic and close up photo of {} ", 
                     "{} in a photo"] 
    elif type =='artist_forget':
        templates = ["A famous artwork by {}",
                     "In the style of {}, a painting", 
                     "a painting in the style of {}", 
                     "{} style painting", 
                     "an art work inspired by {}", 
                     "a beautiful painting drawn by {}", 
                     "a drawing of {}",
                     "{} paintings", 
                     "Image in the style of {}"] 
    elif type == 'NSFW':
        templates = ["An image contained {} woman, indoor",
                     "a {} image, white man, black hair ", 
                     "A {} people", 
                     "a strikingly realistic photo of {}", 
                     "an oil painting of {} woman", 
                     "a photograph of {} image", 
                     "realistic drawing of {} woman",
                     "realistic photo of {} man", 
                     "{} woman in a photo"] 
    else:
        raise ValueError(
            "The current templates don't support the {} type.\n Please insert a new template here".format(type)
        )
    return templates
